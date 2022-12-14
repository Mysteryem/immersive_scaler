import bpy

from typing import Optional, Any


def get_armature() -> Optional[bpy.types.Object]:
    context = bpy.context
    scene = context.scene
    # Get armature from Cats by default if Cats is loaded.
    # Cats stores its currently active armature in an 'armature' EnumProperty added to Scene objects
    # If Cats is loaded, this will always return a string, otherwise, the property won't (shouldn't) exist and None
    # will be returned.
    armature_name = getattr(scene, 'armature', None)
    if armature_name:
        cats_armature = scene.objects.get(armature_name, None)
        if cats_armature and cats_armature.type == 'ARMATURE':
            return cats_armature
        else:
            return None

    # Try to get the armature from the context, this is typically the active object
    obj = context.object
    if obj and obj.type == 'ARMATURE':
        return obj

    # Try to the Object called "Armature"
    armature_name = "Armature"
    obj = scene.objects.get(armature_name, None)
    if obj and obj.type == 'ARMATURE':
        return obj

    # Look through all armature objects, if there's only one, use that
    obj = None
    for o in scene.objects:
        if o.type == 'ARMATURE':
            if obj is None:
                obj = o
            else:
                # There's more than one, we don't know which to use, so return None
                return None
    return obj


if bpy.app.version >= (3, 2):
    # Passing in context_override as a positional-only argument is deprecated as of Blender 3.2, replaced with
    # Context.temp_override
    def op_override(operator, context_override: dict[str, Any], context: Optional[bpy.types.Context] = None,
                    execution_context: Optional[str] = None,
                    undo: Optional[bool] = None, **operator_args) -> set[str]:
        """Call an operator with a context override"""
        args = []
        if execution_context is not None:
            args.append(execution_context)
        if undo is not None:
            args.append(undo)

        if context is None:
            context = bpy.context
        with context.temp_override(**context_override):
            return operator(*args, **operator_args)
else:
    def op_override(operator, context_override: dict[str, Any], context: Optional[bpy.types.Context] = None,
                    execution_context: Optional[str] = None,
                    undo: Optional[bool] = None, **operator_args) -> set[str]:
        """Call an operator with a context override"""
        if context is not None:
            context_base = context.copy()
            context_base.update(context_override)
            context_override = context_base
        args = [context_override]
        if execution_context is not None:
            args.append(execution_context)
        if undo is not None:
            args.append(undo)

        return operator(*args, **operator_args)
