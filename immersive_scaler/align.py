import bpy
import mathutils

# IDK why but this class is necessary
def make_annotations(cls):
    bl_props = {k: v for k, v in cls.__dict__.items() if isinstance(v, tuple)}
    if bl_props:
        if '__annotations__' not in cls.__dict__:
            setattr(cls, '__annotations__', {})
        annotations = cls.__dict__['__annotations__']
        for k, v in bl_props.items():
            annotations[k] = v
            delattr(cls, k)
    return cls

def point_bone(bone, point):
    v1 = (bone.tail - bone.head).normalized()
    v2 = (bone.head - point).normalized()

    # Need to transform the global rotation between the twe vectors
    # into the local space of the bone
    #
    # Essentially, R_l = B @ R_g @ B^-1
    # where
    # R is the desired rotation (rotation_quat_pose)
    #  R_l is the local rotaiton
    #  R_g is the global rotation
    #  B is the bone's global rotation
    #  B^-1 is the inverse of the bone's rotation
    rotation_quat_pose = v1.rotation_difference(v2)
    bm = bone.matrix.to_quaternion()
    bm.rotate(rotation_quat_pose)
    bm.rotate(bone.matrix.inverted())

    bone.rotation_quaternion = bm

def align_bones(dest_bones, root_bone):
    if not root_bone.name in dest_bones:
        return
    rb_length = (root_bone.head - root_bone.tail).length
    rb_target_length = (dest_bones[root_bone.name][0] - dest_bones[root_bone.name][1]).length

    rb_scale = rb_target_length / rb_length
    #rb_translate = dest_bones[root_bone.name][0] - root_bone.head

    m_g = mathutils.Matrix.Translation(dest_bones[root_bone.name][0])
    local_destination = root_bone.matrix.inverted() @ m_g

    #
    # rbm = root_bone.matrix
    # rbm.translation = (0,0,0)
    # root_bone.matrix = rbm

    root_bone.location = local_destination.translation
    #point_bone(root_bone, dest_bones[root_bone.name][1])
    #root_bone.scale = (rb_scale, rb_scale, rb_scale)

    print("Scaling bone %s by a factor of %f"%(root_bone.name, rb_scale))
    print("Moving bone %s from %s to %s"%(root_bone.name, str(root_bone.head), str(dest_bones[root_bone.name][0])))
    for child in root_bone.children:
        # If single child, or tail lines up with a child before
        # moving, scale and point bone to match dest_bones
        align_bones(dest_bones, child)

def align_armature(dest_bones, arm):
    old_active = bpy.context.active_object
    bpy.context.view_layer.objects.active = arm
    bpy.ops.object.mode_set(mode='POSE', toggle = False)
    roots = [b for b in arm.pose.bones if not b.parent]
    for root in roots:
        align_bones(dest_bones, root)

    #bpy.ops.cats_manual.pose_to_rest()
    bpy.context.view_layer.objects.active = old_active


def align_armatures(context):
    print("Selected Objects:")
    for o in  context.selected_objects:
        print(o.name)
    print("Active Object")
    print(context.active_object.name)

    selected_objects = [o for o in context.selected_objects]
    dest_arm = context.active_object
    dest_bones = {}
    bpy.ops.object.mode_set(mode='POSE', toggle = False)
    for bone in dest_arm.pose.bones:
        dest_bones[bone.name] = (bone.head, bone.tail)
    bpy.ops.object.mode_set(mode='POSE', toggle = True)

    for arm in selected_objects:
        if arm == dest_arm:
            continue
        if arm.find_armature() != None:
            arm = arm.find_armature()
        # For every bone, starting at the parent,, move the bone head to match if there is a parallel
        align_armature(dest_bones, arm)



class ArmatureAlign(bpy.types.Operator):
    """Takes one armature and aligns it to another"""
    bl_idname = "armature.align"
    bl_label = "Align Armatures"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        align_armatures(context)
        return{'FINISHED'}

def register():
    bpy.utils.register_class(ArmatureAlign)
    make_annotations(ArmatureAlign)

def unregister():
    bpy.utils.unregister_class(ArmatureAlign)


if __name__ == "__main__":
    register()
