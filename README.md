# FaceSet Creature Rigger (v1.0.0)
**Next-Generation Creature Rigging for Blender 4.2+**

## Quick Start
1. In Blender, go to **Edit > Preferences > Add-ons > Install from Disk...**
2. Select `faceset_creature_rigger.zip` or `faceset_creature_rigger.py`.
3. Check the box to enable **FaceSet Creature Rigger**.
4. In 3D Viewport, press **N** to open the sidebar and navigate to the **FS Rigger** tab.

## Workflow
1. **Sculpt Face Sets**: Enter Sculpt Mode, pick the Face Set brush and paint each creature limb or part (`Ctrl+Click` to start a new set).
2. **Scan**: Click **Scan Parts & Detect Symmetry**. Symmetrical limbs across the X-axis are automatically paired with `.L` and `.R` suffixes.
3. **Configure**: Select roles (`Body`, `FK`, `IK` with Pole target, `Spline IK`, or `Wiggle`).
4. **Generate Rig**: Generates armature with IK pole targets, spline curves, and custom widgets.
5. **Bind**: Generates weight-painted vertex groups with smooth Laplacian passes.
6. **Reset All**: Need to start over? Click **Reset All (Clean Mesh & Rigs)** to cleanly purge generated armatures, vertex groups, modifiers, and curves!
