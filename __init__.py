bl_info = {
    "name": "FaceSet Creature Rigger",
    "author": "FaceSet Creature Rigger Team",
    "version": (1, 0, 0),
    "blender": (4, 2, 0),
    "location": "View3D > Sidebar (N) > FS Rigger",
    "description": "Paint Face Sets (or use materials / loose parts), auto-detect X-symmetry, "
                   "generate bone chains with IK/Pole/Spline/Wiggle controls, bind weights, "
                   "and complete scene reset for creature sculpting.",
    "category": "Rigging",
    "doc_url": "https://github.com",
    "tracker_url": "https://github.com",
}

import json
import math
import re

import bpy
import numpy as np
from bpy.app.handlers import persistent
from bpy.props import (BoolProperty, CollectionProperty, EnumProperty,
                       FloatProperty, IntProperty, PointerProperty,
                       StringProperty)
from bpy.types import Operator, Panel, PropertyGroup, UIList
from mathutils import Matrix, Vector

# --------------------------------------------------------------------------
#  Constants
# --------------------------------------------------------------------------

ROLE_ITEMS = [
    ('BODY', "Body / Spine", "Root body part, FK chain along its longest axis"),
    ('FK', "FK", "Directly posable chain"),
    ('IK', "IK", "IK chain with a target control and optional pole target"),
    ('IKFK', "IK/FK", "IK chain with a FK <-> IK blend slider on the control"),
    ('SPLINE', "Spline IK", "Curve driven chain (tails, tentacles) with hook controls"),
    ('WIGGLE', "Wiggle", "Spring based secondary motion (hair, ears, skirts)"),
]
ROLE_LABEL = {k: v for k, v, _ in ROLE_ITEMS}

SOURCE_ITEMS = [
    ('FACE_SETS', "Sculpt Face Sets", "Use the Face Sets painted in Sculpt Mode"),
    ('MATERIAL', "Material Slots", "Every material slot is one part"),
    ('ISLANDS', "Loose Parts", "Every disconnected mesh island is one part"),
]

PRESET_ITEMS = [
    ('QUADRUPED', "Quadruped Beast", "Spine, Neck, Head, 4 IK/Pole Legs, Tail, Ears"),
    ('BIPED', "Bipedal Character", "Torso, Head, 2 IK/FK Arms with Poles, 2 IK Legs"),
    ('LEVIATHAN', "Leviathan (Tentacles)", "Central body, 8 Spline IK tentacles, Wiggle mantle"),
    ('WINGED', "Winged Creature", "Body, Head, 2 Spline/IK Wings, 2 Legs, Tail"),
    ('ARACHNID', "Arachnid (Multi-Leg)", "Cephalothorax, Abdomen, 8 IK Pole Legs"),
]


def clean_name(s):
    s = re.sub(r'[^\w\-\.]', '_', s).strip()
    return s or "Part"


# --------------------------------------------------------------------------
#  Mesh helpers (numpy based)
# --------------------------------------------------------------------------

def get_topology(me):
    """returns loop_vert, face_of_loop"""
    nl = len(me.loops)
    nf = len(me.polygons)
    loop_vert = np.empty(nl, dtype=np.int32)
    try:
        me.loops.foreach_get("vertex_index", loop_vert)
    except Exception:
        me.attributes[".corner_vert"].data.foreach_get("value", loop_vert)
    total = np.empty(nf, dtype=np.int32)
    try:
        me.polygons.foreach_get("loop_total", total)
    except Exception:
        offs = np.empty(nf + 1, dtype=np.int32)
        me.polygon_offset_indices.foreach_get(offs)
        total = np.diff(offs)
    face_of_loop = np.repeat(np.arange(nf, dtype=np.int32), total)
    return loop_vert, face_of_loop


def get_edges(me):
    e = np.empty(len(me.edges) * 2, dtype=np.int32)
    me.edges.foreach_get("vertices", e)
    return e.reshape(-1, 2)


def world_coords(obj):
    me = obj.data
    co = np.empty(len(me.vertices) * 3, dtype=np.float32)
    me.vertices.foreach_get("co", co)
    co = co.reshape(-1, 3).astype(np.float64)
    mw = np.array(obj.matrix_world, dtype=np.float64)
    return co @ mw[:3, :3].T + mw[:3, 3]


def island_face_keys(me):
    nv = len(me.vertices)
    edges = get_edges(me)
    parent = list(range(nv))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in edges:
        ra, rb = find(int(a)), find(int(b))
        if ra != rb:
            parent[ra] = rb
    roots = np.array([find(i) for i in range(nv)], dtype=np.int32)
    loop_vert, face_of_loop = get_topology(me)
    nf = len(me.polygons)
    keys = np.zeros(nf, dtype=np.int32)
    first = np.full(nf, -1, dtype=np.int64)
    idx = np.arange(len(face_of_loop))
    first[face_of_loop[::-1]] = idx[::-1]
    keys[:] = roots[loop_vert[first]]
    return keys


def get_face_keys(me, source):
    """returns (keys per face, found_flag)"""
    nf = len(me.polygons)
    if source == 'FACE_SETS':
        a = me.attributes.get(".sculpt_face_set")
        if a is None:
            return np.ones(nf, dtype=np.int32), False
        keys = np.empty(nf, dtype=np.int32)
        a.data.foreach_get("value", keys)
        return keys, True
    if source == 'MATERIAL':
        keys = np.empty(nf, dtype=np.int32)
        me.polygons.foreach_get("material_index", keys)
        return keys, True
    return island_face_keys(me), True


# --------------------------------------------------------------------------
#  Geometry & Rigging helpers
# --------------------------------------------------------------------------

def centerline(pts, n, hint=None):
    """Fit an n-bone chain through a point cloud.
    Returns (n+1, 3) array of joint positions, start joint first."""
    c = pts.mean(axis=0)
    X = pts - c
    w, v = np.linalg.eigh(X.T @ X)
    axis = v[:, 2]
    t = X @ axis
    if hint is not None:
        lo = c + axis * t.min()
        hi = c + axis * t.max()
        if np.linalg.norm(lo - hint) > np.linalg.norm(hi - hint):
            axis = -axis
            t = -t
    else:
        flip = False
        if abs(axis[2]) > 0.6:
            flip = axis[2] < 0
        else:
            flip = axis[1] > 0      # walk toward -Y (Blender "front")
        if flip:
            axis = -axis
            t = -t
    tmin, tmax = t.min(), t.max()
    length = max(tmax - tmin, 1e-6)
    hw = max(length / n * 0.75, 1e-6)
    joints = []
    for i in range(n + 1):
        ti = tmin + length * i / n
        base = c + axis * ti
        sel = np.abs(t - ti) <= hw
        if sel.sum() >= 3:
            off = pts[sel].mean(axis=0) - c
            off = off - axis * (off @ axis)
            joints.append(base + off)
        else:
            joints.append(base)
    return np.array(joints)


def resample(joints, count):
    joints = np.asarray(joints)
    seg = np.linalg.norm(np.diff(joints, axis=0), axis=1)
    cum = np.concatenate([[0], np.cumsum(seg)])
    total = cum[-1] if cum[-1] > 0 else 1.0
    out = []
    for k in range(count):
        d = total * k / (count - 1)
        i = min(np.searchsorted(cum, d, side='right') - 1, len(seg) - 1)
        f = (d - cum[i]) / seg[i] if seg[i] > 1e-12 else 0.0
        out.append(joints[i] + (joints[i + 1] - joints[i]) * f)
    return np.array(out)


def calculate_pole_vector(joints, distance_mult=1.2):
    """Computes knee/elbow pole position perpendicular to the limb bending plane."""
    if len(joints) < 3:
        mid = (joints[0] + joints[-1]) * 0.5
        seg = np.linalg.norm(joints[-1] - joints[0])
        return mid + np.array([0.0, -seg * distance_mult, 0.0])

    mid_idx = len(joints) // 2
    root = joints[0]
    mid = joints[mid_idx]
    tip = joints[-1]

    line = tip - root
    line_len2 = float(line @ line)
    if line_len2 > 1e-12:
        proj_t = float((mid - root) @ line) / line_len2
        proj_pt = root + line * proj_t
        bend_dir = mid - proj_pt
        bend_len = float(np.linalg.norm(bend_dir))
    else:
        bend_len = 0.0
        bend_dir = np.zeros(3)

    limb_len = float(np.sum(np.linalg.norm(np.diff(joints, axis=0), axis=1)))
    offset_dist = max(limb_len * 0.45 * distance_mult, 0.05)

    if bend_len > 1e-5:
        bend_dir = bend_dir / bend_len
        pole_pos = mid + bend_dir * offset_dist
    else:
        if mid[0] != 0:
            default_dir = np.array([np.sign(mid[0]) * 0.5, -0.866, 0.0])
        else:
            default_dir = np.array([0.0, -1.0, 0.0])
        default_dir = default_dir / np.linalg.norm(default_dir)
        pole_pos = mid + default_dir * offset_dist

    return pole_pos


def chain_weights(P, heads, tails):
    """Hat-function weights of points P (m,3) along a bone chain -> (m, N)."""
    N = len(heads)
    m = len(P)
    best_d = np.full(m, np.inf)
    best_u = np.zeros(m)
    for j in range(N):
        a, b = heads[j], tails[j]
        ab = b - a
        l2 = max(float(ab @ ab), 1e-12)
        t = np.clip(((P - a) @ ab) / l2, 0.0, 1.0)
        q = a + t[:, None] * ab
        d = ((P - q) ** 2).sum(axis=1)
        better = d < best_d
        best_d[better] = d[better]
        best_u[better] = j + t[better]
    u = np.clip(best_u, 0.5, N - 0.5)
    centers = np.arange(N) + 0.5
    return np.maximum(0.0, 1.0 - np.abs(u[:, None] - centers[None, :]))


def smooth_weights(W, edges, iters, f=0.5):
    if iters <= 0 or len(edges) == 0:
        return W
    cnt = np.zeros(len(W))
    np.add.at(cnt, edges[:, 0], 1)
    np.add.at(cnt, edges[:, 1], 1)
    cnt[cnt == 0] = 1
    for _ in range(iters):
        S = np.zeros_like(W)
        np.add.at(S, edges[:, 0], W[edges[:, 1]])
        np.add.at(S, edges[:, 1], W[edges[:, 0]])
        W = W * (1 - f) + (S / cnt[:, None]) * f
    return W


def limit_influences(W, max_inf, prune=0.01):
    if W.shape[1] > max_inf:
        drop = np.argsort(-W, axis=1)[:, max_inf:]
        np.put_along_axis(W, drop, 0.0, axis=1)
    W[W < prune] = 0.0
    s = W.sum(axis=1, keepdims=True)
    s[s == 0] = 1.0
    return W / s


def seg_dist(p, a, b):
    ab = b - a
    t = max(0.0, min(1.0, (p - a).dot(ab) / max(ab.dot(ab), 1e-12)))
    return (p - (a + ab * t)).length


def vec(a):
    return Vector((float(a[0]), float(a[1]), float(a[2])))


# --------------------------------------------------------------------------
#  Properties
# --------------------------------------------------------------------------

class FSR_Region(PropertyGroup):
    key: IntProperty()
    name: StringProperty(default="Part")
    role: EnumProperty(name="Role", items=ROLE_ITEMS, default='FK')
    bone_count: IntProperty(name="Bones", default=3, min=1, max=64)
    parent_name: StringProperty(name="Parent", description="Part this one attaches to")
    ik_follow_parent: BoolProperty(
        name="IK control follows parent", default=True,
        description="Parent the IK control to the parent part (otherwise to Root)")
    use_pole: BoolProperty(
        name="Add IK Pole Target", default=True,
        description="Generate an elbow or knee pole target control bone")
    spline_ctrls: IntProperty(name="Spline Controls", default=4, min=2, max=10)
    stiffness: FloatProperty(name="Stiffness", default=0.35, min=0.0, max=1.0)
    damping: FloatProperty(name="Damping", default=0.35, min=0.0, max=1.0)
    gravity: FloatProperty(name="Gravity", default=0, min=0.0, max=1.0)
    is_mirrored: BoolProperty(name="Mirrored", default=False)
    mirror_partner: StringProperty(name="Mirror Partner", default="")


class FSR_Settings(PropertyGroup):
    source: EnumProperty(name="Source", items=SOURCE_ITEMS, default='FACE_SETS')
    regions: CollectionProperty(type=FSR_Region)
    active_index: IntProperty()

    # Symmetry options
    auto_symmetry: BoolProperty(
        name="Auto X-Symmetry", default=True,
        description="Automatically detect left/right mirrored parts across the X-axis and name them .L / .R")
    symmetry_tolerance: FloatProperty(
        name="Symmetry Tolerance", default=0.08, min=0.001, max=0.5,
        description="Max deviation factor for X-axis mirror centroid detection")

    # Rig generation options
    add_root: BoolProperty(name="Root Control", default=True)
    use_widgets: BoolProperty(name="Control Shapes", default=True)
    use_pole_targets: BoolProperty(
        name="Generate Pole Targets", default=True,
        description="Create IK pole target bones for knees and elbows")
    pole_distance: FloatProperty(
        name="Pole Distance", default=1.1, min=0.2, max=4.0,
        description="Distance multiplier for pole target projection")

    # Binding options
    smooth_iters: IntProperty(name="Smooth Passes", default=3, min=0, max=30)
    max_influences: IntProperty(name="Max Influences", default=4, min=1, max=8)

    # Preset selection
    preset_choice: EnumProperty(name="Preset", items=PRESET_ITEMS, default='QUADRUPED')


# --------------------------------------------------------------------------
#  Utilities
# --------------------------------------------------------------------------

def find_pair(context):
    """(mesh_obj, armature_obj) for the active selection."""
    ob = context.active_object
    if ob is None:
        return None, None
    if ob.type == 'ARMATURE' and ob.get("fsr_mesh"):
        return bpy.data.objects.get(ob["fsr_mesh"]), ob
    if ob.type == 'MESH':
        for o in bpy.data.objects:
            if o.type == 'ARMATURE' and o.get("fsr_mesh") == ob.name:
                return ob, o
        return ob, None
    return None, None


def ensure_object_mode(context):
    if context.mode != 'OBJECT' and context.active_object is not None:
        bpy.ops.object.mode_set(mode='OBJECT')


def get_circle_widget():
    ob = bpy.data.objects.get("WGT_FSR_Circle")
    if ob:
        return ob
    n = 24
    verts = [(math.cos(2 * math.pi * i / n) * 0.5, 0.0,
              math.sin(2 * math.pi * i / n) * 0.5) for i in range(n)]
    edges = [(i, (i + 1) % n) for i in range(n)]
    me = bpy.data.meshes.new("WGT_FSR_Circle")
    me.from_pydata(verts, edges, [])
    me.update()
    ob = bpy.data.objects.new("WGT_FSR_Circle", me)
    ob.use_fake_user = True
    return ob


def get_sphere_widget():
    ob = bpy.data.objects.get("WGT_FSR_Sphere")
    if ob:
        return ob
    n = 16
    verts = []
    edges = []
    idx = 0
    for plane in ('XY', 'XZ', 'YZ'):
        start = idx
        for i in range(n):
            a = 2 * math.pi * i / n
            ca, sa = math.cos(a) * 0.35, math.sin(a) * 0.35
            if plane == 'XY':
                verts.append((ca, sa, 0.0))
            elif plane == 'XZ':
                verts.append((ca, 0.0, sa))
            else:
                verts.append((0.0, ca, sa))
            if i > 0:
                edges.append((idx - 1, idx))
            idx += 1
        edges.append((idx - 1, start))
    me = bpy.data.meshes.new("WGT_FSR_Sphere")
    me.from_pydata(verts, edges, [])
    me.update()
    ob = bpy.data.objects.new("WGT_FSR_Sphere", me)
    ob.use_fake_user = True
    return ob


def make_bone_collection(arm_data, name):
    try:
        return arm_data.collections.new(name)
    except Exception:
        return None


def order_regions(regs):
    by = {r.name: r for r in regs}
    done, order, pending = set(), [], list(regs)
    changed = True
    while pending and changed:
        changed = False
        for r in list(pending):
            p = r.parent_name
            if (not p) or p not in by or p == r.name or p in done:
                order.append(r)
                done.add(r.name)
                pending.remove(r)
                changed = True
    return order + pending


# --------------------------------------------------------------------------
#  Operators
# --------------------------------------------------------------------------

class FSR_OT_paint(Operator):
    bl_idname = "fsr.paint_face_sets"
    bl_label = "Paint Face Sets"
    bl_description = "Enter Sculpt Mode to paint Face Sets (Face Set brush in the toolbar)"

    @classmethod
    def poll(cls, context):
        return context.active_object and context.active_object.type == 'MESH'

    def execute(self, context):
        if context.mode != 'SCULPT':
            bpy.ops.object.mode_set(mode='SCULPT')
        self.report({'INFO'}, "Pick the 'Face Set' brush and paint your creature parts "
                              "(Ctrl+click = new set). Return to Object Mode and press Scan.")
        return {'FINISHED'}


class FSR_OT_scan(Operator):
    bl_idname = "fsr.scan_regions"
    bl_label = "Scan Parts & Detect Symmetry"
    bl_description = "Analyze the mesh, identify parts, detect X-axis symmetry and infer hierarchy"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return context.active_object and context.active_object.type == 'MESH'

    def execute(self, context):
        ensure_object_mode(context)
        obj = context.active_object
        s = context.scene.fsr
        me = obj.data
        keys, found = get_face_keys(me, s.source)
        if s.source == 'FACE_SETS' and not found:
            self.report({'WARNING'}, "No Face Sets painted yet - entire mesh scanned as a single part.")
        ukeys, inv, counts = np.unique(keys, return_inverse=True, return_counts=True)
        R = len(ukeys)
        if R > 64:
            self.report({'ERROR'}, f"{R} parts found - exceeds limit (max 64).")
            return {'CANCELLED'}

        loop_vert, fol = get_topology(me)
        nv = len(me.vertices)
        M = np.zeros((nv, R), dtype=bool)
        M[loop_vert, inv[fol]] = True
        boundary = M[M.sum(axis=1) > 1].astype(np.int32)
        shared = boundary.T @ boundary if len(boundary) else np.zeros((R, R), dtype=np.int32)

        P = world_coords(obj)
        centroids = np.zeros((R, 3))
        for i in range(R):
            v_idx = np.unique(loop_vert[inv[fol] == i])
            centroids[i] = P[v_idx].mean(axis=0) if len(v_idx) else np.zeros(3)

        body = int(np.argmax(counts))
        dist = np.full(R, 10 ** 6)
        dist[body] = 0
        queue = [body]
        while queue:
            i = queue.pop(0)
            for j in range(R):
                if j != i and shared[i, j] > 0 and dist[j] > dist[i] + 1:
                    dist[j] = dist[i] + 1
                    queue.append(j)

        matched_pairs = {}
        if s.auto_symmetry and R > 1:
            bbox_span = np.ptp(P, axis=0)
            scale = max(float(bbox_span.max()), 1e-4)
            tol = scale * s.symmetry_tolerance
            used_sym = set([body])

            for i in range(R):
                if i in used_sym:
                    continue
                ci = centroids[i]
                if abs(ci[0]) < tol * 0.3:
                    continue

                best_j, best_diff = None, tol
                for j in range(R):
                    if j == i or j in used_sym or j == body:
                        continue
                    cj = centroids[j]
                    dx = abs(ci[0] + cj[0])
                    dy = abs(ci[1] - cj[1])
                    dz = abs(ci[2] - cj[2])
                    diff = math.sqrt(dx * dx + dy * dy + dz * dz)
                    if diff < best_diff:
                        best_diff = diff
                        best_j = j

                if best_j is not None:
                    matched_pairs[i] = best_j
                    matched_pairs[best_j] = i
                    used_sym.add(i)
                    used_sym.add(best_j)

        names = {}
        sym_partner = {}
        limb_counts = {"Arm": 1, "Leg": 1, "Limb": 1}

        processed = set()
        k = 1
        for i in range(R):
            if i in processed:
                continue
            if i == body:
                names[i] = "Body"
                processed.add(i)
            elif i in matched_pairs:
                j = matched_pairs[i]
                ci = centroids[i]
                cb = centroids[body]
                if ci[2] < cb[2]:
                    base_name = f"Leg_{limb_counts['Leg']:02d}"
                    limb_counts['Leg'] += 1
                else:
                    base_name = f"Arm_{limb_counts['Arm']:02d}"
                    limb_counts['Arm'] += 1

                if ci[0] >= 0:
                    names[i] = f"{base_name}.L"
                    names[j] = f"{base_name}.R"
                else:
                    names[i] = f"{base_name}.R"
                    names[j] = f"{base_name}.L"

                sym_partner[i] = names[j]
                sym_partner[j] = names[i]
                processed.add(i)
                processed.add(j)
            else:
                names[i] = f"Part_{k:02d}"
                k += 1
                processed.add(i)

        s.regions.clear()
        for i in range(R):
            r = s.regions.add()
            r.key = int(ukeys[i])
            r.name = names[i]
            if i in sym_partner:
                r.is_mirrored = True
                r.mirror_partner = sym_partner[i]

            if i == body:
                r.role = 'BODY'
                r.bone_count = 4
            else:
                if r.is_mirrored:
                    r.role = 'IK'
                    r.bone_count = 3
                else:
                    r.role = 'FK'
                    r.bone_count = 3

                cands = [j for j in range(R) if j != i and shared[i, j] > 0 and dist[j] < dist[i]]
                if cands:
                    j = max(cands, key=lambda q: shared[i, q])
                    r.parent_name = names[j]
                else:
                    r.parent_name = "Body"

        s.active_index = 0
        pair_count = len(matched_pairs) // 2
        sym_msg = f" with {pair_count} symmetrical pair(s) (.L/.R)" if pair_count > 0 else ""
        self.report({'INFO'}, f"{R} part(s) found{sym_msg}. Adjust roles / bones, then Generate Rig.")
        return {'FINISHED'}


class FSR_OT_select_region(Operator):
    bl_idname = "fsr.select_region"
    bl_label = "Show Part"
    bl_description = "Select the faces of the active part in Edit Mode"

    @classmethod
    def poll(cls, context):
        s = context.scene.fsr
        return (context.active_object and context.active_object.type == 'MESH'
                and 0 <= s.active_index < len(s.regions))

    def execute(self, context):
        ensure_object_mode(context)
        s = context.scene.fsr
        me = context.active_object.data
        keys, _ = get_face_keys(me, s.source)
        sel = (keys == s.regions[s.active_index].key)
        me.polygons.foreach_set("select", sel)
        me.update()
        bpy.ops.object.mode_set(mode='EDIT')
        try:
            bpy.ops.mesh.select_mode(type='FACE')
        except Exception:
            pass
        return {'FINISHED'}


class FSR_OT_generate(Operator):
    bl_idname = "fsr.generate_rig"
    bl_label = "Generate Rig"
    bl_description = "Create an armature with bone chains, IK/Pole targets, Splines and Wiggle physics"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        s = context.scene.fsr
        return (context.active_object and context.active_object.type in {'MESH', 'ARMATURE'}
                and len(s.regions) > 0)

    def execute(self, context):
        mesh_obj, old_arm = find_pair(context)
        if mesh_obj is None:
            self.report({'ERROR'}, "Select the creature mesh first.")
            return {'CANCELLED'}
        ensure_object_mode(context)
        s = context.scene.fsr
        me = mesh_obj.data
        coll = mesh_obj.users_collection[0] if mesh_obj.users_collection else context.collection

        # clean previous rig
        for o in list(bpy.data.objects):
            if o.get("fsr_owner") == mesh_obj.name:
                bpy.data.objects.remove(o, do_unlink=True)

        keys, _ = get_face_keys(me, s.source)
        loop_vert, fol = get_topology(me)
        P = world_coords(mesh_obj)
        region_verts = {}
        for r in s.regions:
            fm = keys == r.key
            if fm.any():
                region_verts[r.key] = np.unique(loop_vert[fm[fol]])

        regions = [r for r in s.regions]
        for r in regions:
            r.name = clean_name(r.name)
        regions = order_regions(regions)

        rig_name = f"{mesh_obj.name}_Rig"
        arm_data = bpy.data.armatures.new(rig_name)
        arm = bpy.data.objects.new(rig_name, arm_data)
        arm["fsr_owner"] = mesh_obj.name
        arm["fsr_mesh"] = mesh_obj.name
        arm["fsr_source"] = s.source
        coll.objects.link(arm)
        arm.show_in_front = True
        arm_data.display_type = 'OCTAHEDRAL'

        for o in context.selected_objects:
            o.select_set(False)
        arm.select_set(True)
        context.view_layer.objects.active = arm
        bpy.ops.object.mode_set(mode='EDIT')
        eb = arm_data.edit_bones

        used = set()

        def uname(base):
            n, i = base, 1
            while n in used:
                i += 1
                n = f"{base}.{i:02d}"
            used.add(n)
            return n

        root_name = None
        allp = P
        dim = float(np.linalg.norm(allp.max(axis=0) - allp.min(axis=0)))
        if s.add_root:
            root_name = uname("Root")
            b = eb.new(root_name)
            cx, cy = allp[:, 0].mean(), allp[:, 1].mean()
            zmin = allp[:, 2].min()
            b.head = Vector((cx, cy, zmin))
            b.tail = Vector((cx, cy, zmin + max(dim * 0.15, 0.05)))
            b.use_deform = False

        info = {}
        ctrl_names = [] if root_name is None else [root_name]
        pole_names = []
        skipped = []

        for r in regions:
            verts = region_verts.get(r.key)
            if verts is None or len(verts) < 3:
                skipped.append(r.name)
                continue
            pts = P[verts]

            parent_bone = root_name
            hint = None
            pinfo = info.get(r.parent_name) if r.parent_name != r.name else None
            if pinfo:
                shared = np.intersect1d(verts, region_verts[pinfo["key"]])
                if len(shared):
                    hint = P[shared].mean(axis=0)
                else:
                    hint = pts.mean(axis=0)

            joints = centerline(pts, r.bone_count, hint)

            if pinfo:
                start = vec(joints[0])
                best, bd = None, 1e18
                for bn in pinfo["bones"]:
                    pb_ = eb[bn]
                    d = seg_dist(start, pb_.head, pb_.tail)
                    if d < bd:
                        best, bd = bn, d
                parent_bone = best

            names = []
            prev = None
            for i in range(r.bone_count):
                nm = uname(f"{r.name}_{i + 1:02d}")
                b = eb.new(nm)
                b.head = vec(joints[i])
                b.tail = vec(joints[i + 1])
                if (b.tail - b.head).length < 1e-3:
                    b.tail = b.head + Vector((0, 0, 1e-3))
                if i == 0:
                    if parent_bone:
                        b.parent = eb[parent_bone]
                else:
                    b.parent = prev
                    b.use_connect = True
                b.use_deform = True
                names.append(nm)
                prev = b

            seg_len = float(np.mean(np.linalg.norm(np.diff(joints, axis=0), axis=1)))
            ctrl = []
            pole = None

            if r.role in {'IK', 'IKFK'}:
                cn = uname(f"IK_{r.name}")
                b = eb.new(cn)
                b.head = vec(joints[-1])
                b.tail = b.head + Vector((0, 0, max(seg_len * 0.6, 0.02)))
                b.use_deform = False
                pn = parent_bone if r.ik_follow_parent else root_name
                if pn:
                    b.parent = eb[pn]
                ctrl.append(cn)

                if s.use_pole_targets and r.use_pole and r.bone_count >= 2:
                    pn_name = uname(f"Pole_{r.name}")
                    pole_pos = calculate_pole_vector(joints, s.pole_distance)
                    pb_pole = eb.new(pn_name)
                    pb_pole.head = vec(pole_pos)
                    pb_pole.tail = pb_pole.head + Vector((0, 0, max(seg_len * 0.4, 0.02)))
                    pb_pole.use_deform = False
                    if parent_bone:
                        pb_pole.parent = eb[parent_bone]
                    ctrl.append(pn_name)
                    pole_names.append(pn_name)
                    pole = pn_name

            elif r.role == 'SPLINE':
                for k, p in enumerate(resample(joints, r.spline_ctrls)):
                    cn = uname(f"{r.name}_CTRL_{k + 1:02d}")
                    b = eb.new(cn)
                    b.head = vec(p)
                    b.tail = b.head + Vector((0, 0, max(seg_len * 0.6, 0.02)))
                    b.use_deform = False
                    if parent_bone:
                        b.parent = eb[parent_bone]
                    ctrl.append(cn)

            ctrl_names += ctrl
            info[r.name] = {"key": r.key, "bones": names, "joints": joints,
                            "ctrl": ctrl, "pole": pole, "region": r}

        bpy.ops.object.mode_set(mode='OBJECT')

        pbs = arm.pose.bones
        widget_circle = get_circle_widget() if s.use_widgets else None
        widget_sphere = get_sphere_widget() if s.use_widgets else None
        col_def = make_bone_collection(arm_data, "Deform")
        col_ctl = make_bone_collection(arm_data, "Controls")
        wiggle_any = False

        for name, d in info.items():
            r = d["region"]
            names, ctrl, pole = d["bones"], d["ctrl"], d["pole"]
            n = len(names)

            if r.role in {'IK', 'IKFK'}:
                c = pbs[names[-1]].constraints.new('IK')
                c.name = "FSR_IK"
                c.target = arm
                c.subtarget = ctrl[0]
                c.chain_count = n

                if pole:
                    c.pole_target = arm
                    c.pole_subtarget = pole
                    c.pole_angle = 0.0

                if r.role == 'IKFK':
                    cp = pbs[ctrl[0]]
                    cp["ik_fk"] = 1.0
                    try:
                        cp.id_properties_ui("ik_fk").update(min=0.0, max=1.0)
                    except Exception:
                        pass
                    fc = c.driver_add("influence")
                    drv = fc.driver
                    drv.type = 'AVERAGE'
                    var = drv.variables.new()
                    var.name = "ikfk"
                    var.type = 'SINGLE_PROP'
                    var.targets[0].id_type = 'OBJECT'
                    var.targets[0].id = arm
                    var.targets[0].data_path = f'pose.bones["{ctrl[0]}"]["ik_fk"]'

            elif r.role == 'SPLINE':
                cu = bpy.data.curves.new(f"{name}_Curve", 'CURVE')
                cu.dimensions = '3D'
                sp = cu.splines.new('NURBS')
                cpts = resample(d["joints"], r.spline_ctrls)
                sp.points.add(len(cpts) - 1)
                for i, p in enumerate(cpts):
                    sp.points[i].co = (float(p[0]), float(p[1]), float(p[2]), 1.0)
                sp.order_u = min(4, len(cpts))
                sp.use_endpoint_u = True
                sp.resolution_u = 12
                cobj = bpy.data.objects.new(f"{name}_Curve", cu)
                cobj["fsr_owner"] = mesh_obj.name
                coll.objects.link(cobj)
                cobj.hide_render = True
                for i, cn in enumerate(ctrl):
                    mod = cobj.modifiers.new(f"Hook_{i}", 'HOOK')
                    mod.object = arm
                    mod.subtarget = cn
                    mod.vertex_indices_set([i])
                    bone = arm_data.bones[cn]
                    mod.matrix_inverse = (arm.matrix_world @ bone.matrix_local).inverted() @ cobj.matrix_world
                c = pbs[names[-1]].constraints.new('SPLINE_IK')
                c.name = "FSR_SplineIK"
                c.target = cobj
                c.chain_count = n
                try:
                    c.y_scale_mode = 'FIT_CURVE'
                    c.xz_scale_mode = 'NONE'
                    c.use_curve_radius = False
                except Exception:
                    pass

            elif r.role == 'WIGGLE':
                wiggle_any = True
                for nm in names:
                    pb = pbs[nm]
                    pb.rotation_mode = 'QUATERNION'
                    pb["fsr_wiggle"] = 1
                    pb["fsr_stiff"] = float(r.stiffness)
                    pb["fsr_damp"] = float(r.damping)
                    pb["fsr_grav"] = float(r.gravity)

            if col_def:
                for nm in names:
                    col_def.assign(arm_data.bones[nm])

        for cn in ctrl_names:
            if col_ctl:
                col_ctl.assign(arm_data.bones[cn])
            if cn in pole_names and widget_sphere:
                pb = pbs[cn]
                pb.custom_shape = widget_sphere
                try:
                    pb.custom_shape_scale_xyz = (1.0, 1.0, 1.0)
                except Exception:
                    pass
            elif widget_circle:
                pb = pbs[cn]
                pb.custom_shape = widget_circle
                try:
                    pb.custom_shape_scale_xyz = (1.2, 1.2, 1.2)
                except Exception:
                    pass

        if wiggle_any:
            arm["fsr_wiggle_on"] = 1
        arm["fsr_regions"] = json.dumps({str(d["key"]): d["bones"] for d in info.values()})

        for o in context.selected_objects:
            o.select_set(False)
        arm.select_set(True)
        context.view_layer.objects.active = arm

        msg = f"Rig generated: {sum(len(d['bones']) for d in info.values())} deform bones."
        if pole_names:
            msg += f" {len(pole_names)} pole target(s) created."
        if skipped:
            msg += f" Skipped (too small): {', '.join(skipped)}."
        self.report({'INFO'}, msg + " Adjust joints in Edit Mode, then Bind.")
        return {'FINISHED'}


class FSR_OT_bind(Operator):
    bl_idname = "fsr.bind"
    bl_label = "Bind / Skin Mesh"
    bl_description = "Create vertex groups with part-guided weights and attach mesh to armature"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        m, a = find_pair(context)
        return m is not None and a is not None

    def execute(self, context):
        obj, arm = find_pair(context)
        ensure_object_mode(context)
        s = context.scene.fsr
        me = obj.data
        source = arm.get("fsr_source", s.source)
        keys, _ = get_face_keys(me, source)
        loop_vert, fol = get_topology(me)
        edges = get_edges(me)
        P = world_coords(obj)
        nv = len(P)
        mw = np.array(arm.matrix_world, dtype=np.float64)

        reg_map = json.loads(arm["fsr_regions"])
        bone_names, spans, verts_of = [], {}, {}
        for k, names in reg_map.items():
            names = [n for n in names if n in arm.data.bones]
            if not names:
                continue
            start = len(bone_names)
            bone_names += names
            spans[k] = (start, len(names), names)
            fm = keys == int(k)
            verts_of[k] = np.unique(loop_vert[fm[fol]]) if fm.any() else np.array([], dtype=np.int64)
        if not bone_names:
            self.report({'ERROR'}, "No deform bones found on the rig.")
            return {'CANCELLED'}

        B = len(bone_names)
        W = np.zeros((nv, B))
        cnt = np.zeros(nv)
        for k, v in verts_of.items():
            cnt[v] += 1
        cnt[cnt == 0] = 1

        for k, (start, N, names) in spans.items():
            v = verts_of[k]
            if len(v) == 0:
                continue
            heads = np.array([(mw @ np.append(np.array(arm.data.bones[n].head_local), 1.0))[:3]
                              for n in names])
            tails = np.array([(mw @ np.append(np.array(arm.data.bones[n].tail_local), 1.0))[:3]
                              for n in names])
            Wr = chain_weights(P[v], heads, tails)
            W[v, start:start + N] += Wr / cnt[v][:, None]

        W = smooth_weights(W, edges, s.smooth_iters)
        W = limit_influences(W, s.max_influences)

        for n in bone_names:
            vg = obj.vertex_groups.get(n)
            if vg:
                obj.vertex_groups.remove(vg)
        for bi, n in enumerate(bone_names):
            vg = obj.vertex_groups.new(name=n)
            for vi in np.nonzero(W[:, bi] > 0.001)[0]:
                vg.add([int(vi)], float(W[vi, bi]), 'REPLACE')

        if obj.parent != arm:
            obj.parent = arm
            obj.parent_type = 'OBJECT'
            obj.matrix_parent_inverse = arm.matrix_world.inverted()
        mod = None
        for m in obj.modifiers:
            if m.type == 'ARMATURE' and m.object == arm:
                mod = m
        if mod is None:
            mod = obj.modifiers.new("FSR Armature", 'ARMATURE')
            mod.object = arm

        for o in context.selected_objects:
            o.select_set(False)
        arm.select_set(True)
        context.view_layer.objects.active = arm
        self.report({'INFO'}, f"Bound {nv} vertices to {B} bones with {s.smooth_iters} smooth passes. Ready for Pose Mode!")
        return {'FINISHED'}


class FSR_OT_unbind(Operator):
    bl_idname = "fsr.unbind"
    bl_label = "Unbind Mesh"
    bl_description = "Detach the mesh from the armature and remove FSR vertex groups without deleting rig"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        m, _ = find_pair(context)
        return m is not None

    def execute(self, context):
        obj, arm = find_pair(context)
        ensure_object_mode(context)

        for m in list(obj.modifiers):
            if m.type == 'ARMATURE' and (arm is None or m.object == arm or m.name == "FSR Armature"):
                obj.modifiers.remove(m)

        if arm and obj.parent == arm:
            mw = obj.matrix_world.copy()
            obj.parent = None
            obj.matrix_world = mw

        if arm and arm.get("fsr_regions"):
            try:
                reg_map = json.loads(arm["fsr_regions"])
                for names in reg_map.values():
                    for n in names:
                        vg = obj.vertex_groups.get(n)
                        if vg:
                            obj.vertex_groups.remove(vg)
            except Exception:
                pass

        self.report({'INFO'}, f"Mesh '{obj.name}' unbound cleanly.")
        return {'FINISHED'}


class FSR_OT_reset_all(Operator):
    bl_idname = "fsr.reset_all"
    bl_label = "Reset All / Deep Purge"
    bl_description = "Complete reset: removes generated rigs, curves, widgets, vertex groups, modifiers and restores pristine mesh"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return context.active_object is not None

    def invoke(self, context, event):
        return context.window_manager.invoke_confirm(self, event)

    def execute(self, context):
        mesh_obj, arm = find_pair(context)
        ensure_object_mode(context)
        s = context.scene.fsr
        target_name = mesh_obj.name if mesh_obj else ""

        deleted_objs = 0
        deleted_vgroups = 0

        # 1. Unbind mesh & purge vertex groups
        if mesh_obj:
            for m in list(mesh_obj.modifiers):
                if m.type == 'ARMATURE' and (m.name.startswith("FSR") or (arm and m.object == arm)):
                    mesh_obj.modifiers.remove(m)

            if arm and mesh_obj.parent == arm:
                mw = mesh_obj.matrix_world.copy()
                mesh_obj.parent = None
                mesh_obj.matrix_world = mw

            if arm and arm.get("fsr_regions"):
                try:
                    reg_map = json.loads(arm["fsr_regions"])
                    for names in reg_map.values():
                        for n in names:
                            vg = mesh_obj.vertex_groups.get(n)
                            if vg:
                                mesh_obj.vertex_groups.remove(vg)
                                deleted_vgroups += 1
                except Exception:
                    pass

        # 2. Delete generated Armatures, Spline Curves, Hook objects
        for o in list(bpy.data.objects):
            if (target_name and o.get("fsr_owner") == target_name) or o == arm:
                data = o.data
                bpy.data.objects.remove(o, do_unlink=True)
                deleted_objs += 1
                if data and data.users == 0:
                    if isinstance(data, bpy.types.Armature):
                        bpy.data.armatures.remove(data)
                    elif isinstance(data, bpy.types.Curve):
                        bpy.data.curves.remove(data)

        # 3. Clean widget objects if no other rigs exist
        any_rigs = any(o.type == 'ARMATURE' and o.get("fsr_mesh") for o in bpy.data.objects)
        if not any_rigs:
            for wgt_name in ("WGT_FSR_Circle", "WGT_FSR_Sphere"):
                w = bpy.data.objects.get(wgt_name)
                if w:
                    wdata = w.data
                    bpy.data.objects.remove(w, do_unlink=True)
                    if wdata and wdata.users == 0:
                        bpy.data.meshes.remove(wdata)

        # 4. Clear scene regions and reset properties
        s.regions.clear()
        s.active_index = 0

        # 5. Clear simulation state
        _wig_state.clear()
        _wig_last.clear()

        self.report({'INFO'}, f"Complete Reset: {deleted_objs} object(s), {deleted_vgroups} vertex group(s) purged. All reset.")
        return {'FINISHED'}


class FSR_OT_clear_pose(Operator):
    bl_idname = "fsr.clear_pose"
    bl_label = "Clear Pose"
    bl_description = "Reset all bone transforms of the rig back to rest position"

    @classmethod
    def poll(cls, context):
        return find_pair(context)[1] is not None

    def execute(self, context):
        arm = find_pair(context)[1]
        for pb in arm.pose.bones:
            pb.location = (0, 0, 0)
            pb.scale = (1, 1, 1)
            pb.rotation_quaternion = (1, 0, 0, 0)
            pb.rotation_euler = (0, 0, 0)
        self.report({'INFO'}, "Bone transforms cleared to rest pose.")
        return {'FINISHED'}


class FSR_OT_apply_preset(Operator):
    bl_idname = "fsr.apply_preset"
    bl_label = "Apply Creature Preset"
    bl_description = "Configure typical part roles and bone chain counts for selected creature type"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        s = context.scene.fsr
        return len(s.regions) > 0

    def execute(self, context):
        s = context.scene.fsr
        p = s.preset_choice

        for r in s.regions:
            name_lower = r.name.lower()
            if "body" in name_lower or "spine" in name_lower or "torso" in name_lower:
                r.role = 'BODY'
                r.bone_count = 5 if p in {'QUADRUPED', 'WINGED'} else 4
            elif "leg" in name_lower:
                r.role = 'IK'
                r.bone_count = 3
                r.use_pole = True
            elif "arm" in name_lower:
                r.role = 'IKFK' if p == 'BIPED' else 'IK'
                r.bone_count = 3
                r.use_pole = True
            elif "tail" in name_lower or "tentacle" in name_lower:
                r.role = 'SPLINE'
                r.bone_count = 6
                r.spline_ctrls = 5
            elif "ear" in name_lower or "hair" in name_lower or "skirt" in name_lower or "jaw" in name_lower:
                r.role = 'WIGGLE'
                r.bone_count = 2
                r.stiffness = 0.35
                r.damping = 0.4
            elif "wing" in name_lower:
                r.role = 'FK'
                r.bone_count = 4

        self.report({'INFO'}, f"Applied '{dict(PRESET_ITEMS).get(p, p)}' blueprint to active parts.")
        return {'FINISHED'}


class FSR_OT_bake_wiggle(Operator):
    bl_idname = "fsr.bake_wiggle"
    bl_label = "Bake Wiggle to Keyframes"
    bl_description = "Simulate the timeline frame range and bake wiggle physics into bone keyframes"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        a = find_pair(context)[1]
        return a is not None and any(pb.get("fsr_wiggle") for pb in a.pose.bones)

    def execute(self, context):
        arm = find_pair(context)[1]
        scn = context.scene
        arm["fsr_wiggle_on"] = 1
        wig = [pb for pb in arm.pose.bones if pb.get("fsr_wiggle")]
        start, end = scn.frame_start, scn.frame_end
        _wig_last.pop(scn.name, None)
        _wig_state.clear()
        for f in range(start, end + 1):
            scn.frame_set(f)
            for pb in wig:
                pb.keyframe_insert("rotation_quaternion", frame=f)
        arm["fsr_wiggle_on"] = 0
        scn.frame_set(start)
        self.report({'INFO'}, f"Baked wiggle for {len(wig)} bones ({start}-{end}). Live physics paused.")
        return {'FINISHED'}


# --------------------------------------------------------------------------
#  Wiggle physics simulation (frame change handler)
# --------------------------------------------------------------------------

_wig_state = {}
_wig_last = {}


def _depth(pb):
    d = 0
    while pb.parent:
        pb = pb.parent
        d += 1
    return d


def wiggle_step(scene, arm, steps, reset):
    fps = scene.render.fps / scene.render.fps_base
    mw = arm.matrix_world
    mw_inv = mw.inverted()
    g = Vector((0.0, 0.0, -9.81))
    pbs = [pb for pb in arm.pose.bones if pb.get("fsr_wiggle")]
    pbs.sort(key=_depth)
    sub = max(1, steps * 3)
    h = (steps / fps) / sub if steps > 0 else 0.0

    for pb in pbs:
        pb.matrix_basis = Matrix()
        if pb.parent:
            M = pb.parent.matrix @ pb.parent.bone.matrix_local.inverted() @ pb.bone.matrix_local
        else:
            M = pb.bone.matrix_local.copy()
        Mw = mw @ M
        head = Mw.translation.copy()
        rest_tail = Mw @ Vector((0.0, pb.bone.length, 0.0))
        length = (rest_tail - head).length
        key = (arm.name, pb.name)
        st = _wig_state.get(key)
        if reset or st is None or length < 1e-9:
            _wig_state[key] = {"pos": rest_tail.copy(), "vel": Vector((0.0, 0.0, 0.0))}
            pb.matrix = mw_inv @ Mw
            continue

        stiff = float(pb.get("fsr_stiff", 0.35))
        damp = float(pb.get("fsr_damp", 0.35))
        grav = float(pb.get("fsr_grav", 0.0))
        omega = 4.0 + stiff * 40.0
        k = omega * omega
        c = damp * 20.0
        pos, vel = st["pos"], st["vel"]
        for _ in range(sub):
            acc = (rest_tail - pos) * k + g * grav
            vel = (vel + acc * h) * math.exp(-c * h)
            pos = pos + vel * h

        d = pos - head
        if d.length < 1e-9:
            d = rest_tail - head
        d.normalize()
        pos = head + d * length
        st["pos"], st["vel"] = pos, vel

        y_rest = (rest_tail - head).normalized()
        R = y_rest.rotation_difference(d).to_matrix().to_4x4()
        Mw_new = Matrix.Translation(head) @ R @ Matrix.Translation(-head) @ Mw
        pb.matrix = mw_inv @ Mw_new


@persistent
def fsr_frame_handler(scene, depsgraph=None):
    arms = [o for o in scene.objects if o.type == 'ARMATURE' and o.get("fsr_wiggle_on")]
    if not arms:
        return
    frame = scene.frame_current
    last = _wig_last.get(scene.name)
    if last is None or frame <= last or frame - last > 4:
        reset, steps = True, 0
    else:
        reset, steps = False, frame - last
    for arm in arms:
        try:
            wiggle_step(scene, arm, steps, reset)
        except Exception as e:
            print("FSR wiggle error:", e)
    _wig_last[scene.name] = frame


@persistent
def fsr_load_post(_):
    _wig_state.clear()
    _wig_last.clear()


# --------------------------------------------------------------------------
#  UI & Panels
# --------------------------------------------------------------------------

class FSR_UL_regions(UIList):
    def draw_item(self, context, layout, data, item, icon, active_data, active_propname, index):
        row = layout.row(align=True)
        if item.is_mirrored:
            row.label(text="", icon='MOD_MIRROR')
        else:
            row.label(text="", icon='FACESEL')
        row.prop(item, "name", text="", emboss=False)
        row.label(text=ROLE_LABEL.get(item.role, item.role))
        row.label(text=f"{item.bone_count}b")


class FSR_PT_main(Panel):
    bl_label = "FaceSet Creature Rigger"
    bl_idname = "FSR_PT_main"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "FS Rigger"

    def draw(self, context):
        lay = self.layout
        s = context.scene.fsr

        # Box 1: Define Parts & Symmetry
        box = lay.box()
        box.label(text="1. Define Parts & Symmetry", icon='BRUSH_DATA')
        box.prop(s, "source", text="")
        if s.source == 'FACE_SETS':
            box.operator("fsr.paint_face_sets", icon='SCULPTMODE_HLT')

        # Symmetry row
        sym_box = box.box()
        sym_row = sym_box.row()
        sym_row.prop(s, "auto_symmetry", icon='MOD_MIRROR')
        if s.auto_symmetry:
            sym_box.prop(s, "symmetry_tolerance", slider=True)

        box.operator("fsr.scan_regions", icon='VIEWZOOM')

        # Box 2: Roles, Bones & Presets
        box = lay.box()
        box.label(text="2. Roles & Bone Chains", icon='GROUP_BONE')

        # Creature Presets
        p_row = box.row(align=True)
        p_row.prop(s, "preset_choice", text="")
        p_row.operator("fsr.apply_preset", text="Apply", icon='PRESET')

        box.template_list("FSR_UL_regions", "", s, "regions", s, "active_index", rows=5)
        if 0 <= s.active_index < len(s.regions):
            r = s.regions[s.active_index]
            col = box.column(align=True)
            col.prop(r, "role")
            col.prop(r, "bone_count")
            col.prop_search(r, "parent_name", s, "regions", icon='BONE_DATA')
            if r.role in {'IK', 'IKFK'}:
                col.prop(r, "ik_follow_parent")
                col.prop(r, "use_pole")
            elif r.role == 'SPLINE':
                col.prop(r, "spline_ctrls")
            elif r.role == 'WIGGLE':
                col.prop(r, "stiffness", slider=True)
                col.prop(r, "damping", slider=True)
                col.prop(r, "gravity", slider=True)
            box.operator("fsr.select_region", icon='RESTRICT_SELECT_OFF')

        # Box 3: Generate
        box = lay.box()
        box.label(text="3. Rig Generation", icon='ARMATURE_DATA')
        row = box.row()
        row.prop(s, "add_root")
        row.prop(s, "use_widgets")
        pole_row = box.row()
        pole_row.prop(s, "use_pole_targets")
        if s.use_pole_targets:
            pole_row.prop(s, "pole_distance")
        box.operator("fsr.generate_rig", icon='BONE_DATA')

        # Box 4: Skinning & Bind
        box = lay.box()
        box.label(text="4. Skinning & Bind", icon='MOD_ARMATURE')
        row = box.row()
        row.prop(s, "smooth_iters")
        row.prop(s, "max_influences")
        bind_row = box.row(align=True)
        bind_row.operator("fsr.bind", icon='GROUP_VERTEX')
        bind_row.operator("fsr.unbind", icon='X')
        box.operator("fsr.clear_pose", icon='LOOP_BACK')

        # Wiggle simulation box
        m, a = find_pair(context)
        if a is not None and a.get("fsr_wiggle_on") is not None:
            wbox = lay.box()
            wbox.label(text="Wiggle Simulation", icon='FORCE_HARMONIC')
            wbox.prop(a, '["fsr_wiggle_on"]', text="Live Wiggle (Timeline)")
            wbox.operator("fsr.bake_wiggle", icon='REC')

        # Box 5: Deep Reset / Purge
        rbox = lay.box()
        rbox.label(text="Danger Zone / Complete Purge", icon='CANCEL')
        rbox.operator("fsr.reset_all", text="Reset All (Clean Mesh & Rigs)", icon='TRASH')


# --------------------------------------------------------------------------
#  Registration
# --------------------------------------------------------------------------

classes = (
    FSR_Region, FSR_Settings,
    FSR_OT_paint, FSR_OT_scan, FSR_OT_select_region, FSR_OT_generate,
    FSR_OT_bind, FSR_OT_unbind, FSR_OT_reset_all,
    FSR_OT_clear_pose, FSR_OT_apply_preset, FSR_OT_bake_wiggle,
    FSR_UL_regions, FSR_PT_main,
)


def register():
    for c in classes:
        bpy.utils.register_class(c)
    bpy.types.Scene.fsr = PointerProperty(type=FSR_Settings)
    if fsr_frame_handler not in bpy.app.handlers.frame_change_post:
        bpy.app.handlers.frame_change_post.append(fsr_frame_handler)
    if fsr_load_post not in bpy.app.handlers.load_post:
        bpy.app.handlers.load_post.append(fsr_load_post)


def unregister():
    if fsr_frame_handler in bpy.app.handlers.frame_change_post:
        bpy.app.handlers.frame_change_post.remove(fsr_frame_handler)
    if fsr_load_post in bpy.app.handlers.load_post:
        bpy.app.handlers.load_post.remove(fsr_load_post)
    del bpy.types.Scene.fsr
    for c in reversed(classes):
        bpy.utils.unregister_class(c)


if __name__ == "__main__":
    register()
