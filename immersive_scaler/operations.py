import bpy
import mathutils
import math
import numpy as np
from typing import cast

from .common import get_armature, op_override

def obj_in_scene(obj):
    for o in bpy.context.view_layer.objects:
        if o is obj:
            return True
    return False

def get_body_meshes():
    arm = get_armature()
    meshes = []
    for c in arm.children:
        if not obj_in_scene(c):
            continue
        if len(c.users_scene) == 0:
            continue
        if c.type == 'MESH':
            meshes.append(c)
    return meshes


def unhide_obj(obj):
    # FIXME: completely broken, unhide_obj is this function
    if not 'hide_states' in dir(unhide_obj):
        unhide_obj.hide_states = {}
    if not obj in unhide_obj.hide_states:
        print("Storing hide state of {} as {}".format(obj.name, obj.hide_get()))
        unhide_obj.hide_states[obj] = obj.hide_get()
    obj.hide_set(False)


def rehide_obj(obj):
    # FIXME: completely broken, unhide_obj is the previous function
    if not 'hide_states' in dir(unhide_obj):
        return
    if not obj in unhide_obj.hide_states:
        return
    print("Setting hide state of {} to {}".format(obj.name, unhide_obj.hide_states[obj]))
    obj.hide_set(unhide_obj.hide_states[obj])
    del(unhide_obj.hide_states[obj])


bone_names = {
    "right_shoulder": ["rightshoulder", "shoulderr", "rshoulder"],
    "right_arm": ["rightarm", "armr", "rarm", "upperarmr", "rightupperarm"],
    "right_elbow": ["rightelbow", "elbowr", "relbow", "lowerarmr", "rightlowerarm", "lowerarmr", "forearmr"],
    "right_wrist": ["rightwrist", "wristr", "rwrist", "handr", "righthand"],
    "right_leg": ["rightleg", "legr", "rleg", "upperlegr", "thighr","rightupperleg"],
    "right_knee": ["rightknee", "kneer", "rknee", "lowerlegr", "calfr", "rightlowerleg", "shinr"],
    "right_ankle": ["rightankle", "ankler", "rankle", "rightfoot", "footr", "rightfoot"],
    "right_eye": ['eyer', 'righteye', 'eyeright', 'righteye001'],
    "left_shoulder": ["leftshoulder", "shoulderl", "lshoulder"],
    "left_arm": ["leftarm", "arml", "larm", "upperarml", "leftupperarm"],
    "left_elbow": ["leftelbow", "elbowl", "lelbow", "lowerarml", "leftlowerarm", "lowerarml", "forearml"],
    "left_wrist": ["leftwrist", "wristl", "rwrist", "handl", "lefthand"],
    "left_leg": ["leftleg", "legl", "rleg", "upperlegl", "thighl","leftupperleg"],
    "left_knee": ["leftknee", "kneel", "lknee", "lowerlegl", "calfl", "shinl", "leftlowerleg"],
    "left_ankle": ["leftankle", "anklel", "lankle", "leftfoot", "footl", "leftfoot"],
    "left_eye": ['eyel', 'lefteye', 'eyeleft', 'lefteye001'],
    "head": ["head"],
    "neck": ["neck"],
}

def get_bone(name, arm):
    # First check that there's no override
    s = bpy.context.scene
    override = getattr(s, "override_" + name)
    if override != '_None':
        return arm.pose.bones[override]
    name_list = bone_names[name]
    bone_lookup = dict([(bone.name.lower().translate(dict.fromkeys(map(ord, u" _."))), bone) for bone in arm.pose.bones])
    for n in name_list:
        if n in bone_lookup:
            return bone_lookup[n]
    return arm.pose.bones[name]


def bound_box_to_co_array(obj: bpy.types.Object):
    # Note that bounding boxes of objects correspond to the object with shape keys and modifiers applied
    # Bounding boxes are 2D bpy_prop_array, each bounding box is represented by 8 (x, y, z) rows. Since this is a
    # bpy_prop_array, the dtype must match the internal C type, otherwise an error is raised.
    bb_co = np.empty((8, 3), dtype=np.single)

    # Temporarily disabling modifiers to get a more accurate bounding box of the mesh and then re-enabling the modifiers
    # would be far too performance heavy. Changing active shape key might be too heavy too. Though, even if we change
    # the active shape key or modifiers in code, the bounding box doesn't seem to update right away.
    obj.bound_box.foreach_get(bb_co)

    return bb_co


def get_global_z_from_co_ndarray(v_co: np.ndarray, wm: mathutils.Matrix):
    if v_co.dtype != np.single:
        # The dtype isn't too important when not using foreach_set/foreach_get. Given another float type it would just
        # mean we're doing math with more (or less) precision than existed in the first place.
        raise ValueError("co array should be single precision float")
    if len(wm.row) != 4 or len(wm.col) != 4:
        raise ValueError("matrix must be 4x4")
    # Create a view of the array so that when we set the shape, we set the shape of the view rather than the original
    # array
    v_co = v_co.view()
    # Convert mathutils.Matrix to an np.ndarray
    wm = np.array(wm, dtype=np.single)

    # Change the shape we view the data with so that each element corresponds to a single vertex's (x,y,z)
    v_co.shape = (-1, 3)
    # We have a 4x4 matrix, but each vertex co has only 3 elements. Unlike multiplying a 4x4 mathutils.Matrix
    # with a 3-length mathutils.Vector, numpy won't automatically extend the vector to have a 4th element with value
    # of 1 (and will instead raise an error).
    # Add 1 to the end of each element in the array using np.insert
    index_to_insert_before = 3
    value_to_insert = 1
    v_co4 = np.insert(v_co, index_to_insert_before, value_to_insert, axis=1)
    # To multiply the matrix (4, 4) by each vector in (num_verts, 4), we can transpose the entire array to turn it
    # on its side and treat it as one giant matrix whereby each column is one vector. Note that with numpy, the
    # transpose of an ndarray is a view, no data is copied.
    # ┌a, b, c, d┐   ┌x1, y1, z1, 1┐    ┌a, b, c, d┐   ┌x1, x2, x3, x4, …, xn┐
    # │e, f, g, h│ ? │x2, y2, z2, 1│ -> │e, f, g, h│ @ │y1, y2, y3, y4, …, yn│
    # │i, j, k, l│   │x3, y3, z3, 1│    │i, j, k, l│   │z1, z2, z3, z4, …, zn│
    # └m, n, o, p┘   │x4, y4, z4, 1│    └m, n, o, p┘   └ 1,  1,  1,  1, …,  1┘
    #                ┊ …,  …,  …, …┊
    #                └xn, yn, zn, 1┘
    # This gives us a result with the shape (4, num_verts). The alternative would be to transpose the matrix instead
    # and do `vco_4 @ wm.T`, which would give us the transpose of the first result with the shape (num_verts, 4).
    v_co4_global_t = wm @ v_co4.T
    # We only care about the z values, which will all currently be in index 2
    global_z_only = v_co4_global_t[2]
    return global_z_only


def get_lowest_point():
    arm = get_armature()
    bones = set()
    for bone in (get_bone("left_ankle", arm), get_bone("right_ankle", arm)):
        bones.add(bone.name)
        bones.update(b.name for b in bone.children_recursive)

    meshes = []
    for o in get_body_meshes():
        # Get z components of worldspace bounding box
        global_bb_z = get_global_z_from_co_ndarray(bound_box_to_co_array(o), o.matrix_world)
        # Get minimum z component. This is exceedingly likely to be lower or the same as the lowest vertex in the mesh.
        likely_lowest_possible_vertex_z = np.min(global_bb_z)
        # Add the minimum z component along with the mesh object
        meshes.append((likely_lowest_possible_vertex_z, o))
    # Sort meshes by lowest bounding box first, that way, we can stop checking meshes once we get to a mesh whose lowest
    # corner of the bounding box is higher than the current lowest vertex
    meshes.sort(key=lambda t: t[0])

    lowest_vertex_z = math.inf
    lowest_foot_z = math.inf

    for likely_lowest_possible_vertex_z, o in meshes:
        mesh = o.data
        if not mesh.vertices:
            # Immediately skip if there's no vertices
            continue

        found_feet_previously = lowest_foot_z < math.inf
        current_min = lowest_foot_z if found_feet_previously else lowest_vertex_z
        if likely_lowest_possible_vertex_z > current_min:
            # Lowest possible vertex of this mesh is exceedingly likely to be higher than the current lowest found.
            # Since the meshes are sorted by lowest possible vertex first, any subsequent meshes will be the same, so we
            # don't need to check them.
            break

        if mesh.shape_keys:
            # Exiting edit mode synchronizes a mesh's vertex and 'basis' (reference) shape key positions, but if one of
            # them is modified outside of edit mode without the other being modified in the same way also, the two can
            # become desynchronized. What users see in the 3D view corresponds to the reference shape key, so we'll
            # assume that has the correct positions.
            num_verts = len(mesh.vertices)
            # vertex positions ('co') are (x,y,z) vectors, but get flattened when using foreach_get/set, so the
            # resulting array is 3 times the number of vertices
            v_co = np.empty(num_verts * 3, dtype=np.single)
            # Directly copy the 'co' of the reference shape key into the v_cos array (type must match the internal C
            # type for a direct copy)
            mesh.shape_keys.reference_key.data.foreach_get('co', v_co)
            # Directly paste the 'co' copied from the reference shape key into the 'co' of the vertices
            mesh.vertices.foreach_set('co', v_co)
        else:
            v_co = None

        foot_group_indices = {idx for idx, vg in enumerate(o.vertex_groups) if vg.name in bones}
        wm = o.matrix_world
        if not foot_group_indices:
            # Don't need to get vertex weights, so we can use numpy for performance
            # If the mesh had shape keys, we will already have the v_co array, otherwise, get it from the vertices
            if v_co is None:
                num_verts = len(mesh.vertices)
                v_co = np.empty(num_verts * 3, dtype=np.single)
                mesh.vertices.foreach_get('co', v_co)
            global_z_only = get_global_z_from_co_ndarray(v_co, wm)
            # Get the minimum value
            min_global_z = np.min(global_z_only)
            # Compare against the current lowest vertex z and set it to whichever is smallest
            lowest_vertex_z = min(lowest_vertex_z, min_global_z)
        else:
            # There are unfortunately no fast methods for getting all vertex weights, so we must resort to iteration
            # Helper function to reduce duplicate code
            def find_lowest_z_in_ankles(foot_z_list, vertices):
                for vert in vertices:
                    for group in vert.groups:
                        # .group is the index of the vertex_group
                        if group.group in foot_group_indices and group.weight:
                            # The current vertex is weighted
                            # Calculate the global (world) position
                            world_co = wm @ vert.co
                            # Append the z component
                            foot_z_list.append(world_co[2])
                            # Don't need to check any of the remaining vertex groups, so break the inner loop
                            break
                return min(foot_z_list)
            if found_feet_previously:
                # We already have a value for lowest_foot_z, so we won't be using lowest_vertex_z and only need to care
                # about vertices that are weighted to the ankles or below
                lowest_foot_z = find_lowest_z_in_ankles([lowest_foot_z], mesh.vertices)
            else:
                # We don't have a value for lowest_foot_z yet, so we need to record the lowest vertices even if they're
                # not weighted to the ankles or below
                vertex_z = [lowest_vertex_z]
                # Using an iterator specifically because we may want to change to a more optimised loop part way through
                v_it = iter(mesh.vertices)
                found_feet = False
                for v in v_it:
                    wco = wm @ v.co
                    z = wco[2]
                    vertex_z.append(z)
                    # Check if v is weighted to the ankle or a child
                    for g in v.groups:
                        if g.group in foot_group_indices and g.weight:
                            found_feet = True
                            # lowest_vertex_z is irrelevant now that we've found a vertex belonging to feet
                            # Continue iterating with a slightly more optimised loop until the iterator is exhausted
                            lowest_foot_z = find_lowest_z_in_ankles([lowest_foot_z, z], v_it)
                            break
                if not found_feet:
                    # Didn't manage to find any vertices belonging to feet
                    lowest_vertex_z = min(vertex_z)
    if lowest_foot_z == math.inf:
        if lowest_vertex_z == math.inf:
            raise RuntimeError("No mesh data found")
        else:
            return lowest_vertex_z
    return lowest_foot_z


def get_highest_point():
    # Almost the same as get_lowest_point for obvious reasons, but only using numpy since we don't need to check vertex
    # weights
    meshes = []
    for o in get_body_meshes():
        # Get z components of worldspace bounding box
        global_bb_z = get_global_z_from_co_ndarray(bound_box_to_co_array(o), o.matrix_world)
        # Get minimum z component. This is exceedingly likely to be lower or the same as the lowest vertex in the mesh.
        likely_lowest_possible_vertex_z = np.min(global_bb_z)
        # Add the minimum z component along with the mesh object
        meshes.append((likely_lowest_possible_vertex_z, o))
    # Sort meshes by highest bounding box first, that way, we can stop checking meshes once we get to a mesh whose
    # highest corner of the bounding box is lower than the current highest vertex
    meshes.sort(key=lambda t: t[0], reverse=True)

    minimum_value = -math.inf
    highest_vertex_z = minimum_value
    for likely_highest_possible_vertex_z, o in meshes:
        wm = o.matrix_world
        mesh = o.data

        # Sometimes the 'basis' (reference) shape key and mesh vertices can become desynchronized. If a mesh has shape
        # keys, then the reference shape key is what users will see in Blender, so get vertex positions from that.
        vertices = mesh.shape_keys.reference_key.data if mesh.shape_keys else mesh.vertices
        num_verts = len(vertices)
        if num_verts == 0:
            continue

        if likely_highest_possible_vertex_z < highest_vertex_z:
            # Highest possible vertex of this mesh is exceedingly likely to be lower than the current highest found.
            # Since the meshes are sorted by highest possible vertex first, any subsequent meshes will be the same, so
            # we don't need to check them.
            break

        v_co = np.empty(num_verts * 3, dtype=np.single)
        vertices.foreach_get('co', v_co)
        # Get global vertex z values
        global_z_only = get_global_z_from_co_ndarray(v_co, wm)
        # Get the maximum value
        max_global_z = np.max(global_z_only)
        # Compare against the current highest vertex z and set it to whichever is greatest
        highest_vertex_z = max(highest_vertex_z, max_global_z)
    if highest_vertex_z == minimum_value:
        raise RuntimeError("No mesh data found")
    else:
        return highest_vertex_z


def get_view_y(obj, custom_scale_ratio=.4537):
    # VRC uses the distance between the head bone and right hand in
    # t-pose as the basis for world scale. Enforce t-pose locally to
    # grab this number
    unhide_obj(obj)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.mode_set(mode='POSE', toggle = False)

    # Gets the in-vrchat virtual height that the view will be at,
    # relative to your actual floor.

    # Magic that somebody posted in discord. I'm going to just assume
    # these constants are correct. Testing shows it's at least pretty
    # darn close
    view_y = (head_to_hand(obj) / custom_scale_ratio) + .005

    bpy.ops.object.mode_set(mode='POSE', toggle = True)

    return view_y

def get_current_scaling(obj):

    unhide_obj(obj)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.mode_set(mode='POSE', toggle = False)

    ratio = head_to_hand(obj) / (get_eye_height(obj) - .005)

    bpy.ops.object.mode_set(mode='POSE', toggle = True)
    return ratio


def head_to_hand(obj):
    """Get the length from the head to the start of the wrist bone as if the armature was in t-pose"""
    # Since arms might not be flat, add the length of the arm to the x
    # coordinate of the upper arm
    headpos = get_bone("head", obj).head
    neckpos = get_bone("neck", obj).head
    upper_arm = get_bone("right_arm", obj).head
    elbow = get_bone("right_elbow", obj).head
    wrist = get_bone("right_wrist", obj).head
    # Unity bones are from joint to joint, ignoring whatever the tail may be in Blender
    # Length from upper_arm joint to elbow joint
    upper_arm_length = (upper_arm - elbow).length
    # Length from elbow joint to wrist joint
    lower_arm_length = (elbow - wrist).length
    arm_length = upper_arm_length + lower_arm_length
    # We're working with the right arm, which is on the -x side in Blender, so subtract arm length from the x of start
    # of the arm to get the position of the wrist if the arms were in t-pose (we're assuming a t-pose without any
    # shoulder movement).
    t_hand_pos = mathutils.Vector((upper_arm.x - arm_length, upper_arm.y, upper_arm.z))
    bpy.context.scene.cursor.location = t_hand_pos
    return (headpos - t_hand_pos).length


def calculate_arm_rescaling(obj, head_arm_change):
    # Calculates the percent change in arm length needed to create a
    # given change in head-hand length.

    unhide_obj(obj)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.mode_set(mode='POSE', toggle = False)

    rhandpos = get_bone("right_wrist", obj).head
    rarmpos = get_bone("right_arm", obj).head
    headpos = get_bone("head", obj).head
    neckpos = get_bone("neck", obj).head

    # Reset t-pose to whatever it was before since we have the data we
    # need
    bpy.ops.object.mode_set(mode='POSE', toggle = True)

    total_length = head_to_hand(obj)
    print("Arm length is {}".format(total_length))
    arm_length = (rarmpos - rhandpos).length
    neck_length = abs((headpos[2] - rarmpos[2]))

    # Sanity check - compare the difference between head_to_hand and manual
    # print("")
    # print("-------head_to_hand: %f" %total_length)
    # print("-------manual, assuming t-pose: %f" %(headpos - rhandpos).length)
    # print("")

    # Also derived using sympy. See below.
    shoulder_length = math.sqrt((total_length - neck_length) * (total_length + neck_length)) - arm_length

    # funky equation for all this - derived with sympy:
    # solveset(Eq(a * x, sqrt((c * b + s)**2 + y**2)), b)
    # where
    # x is total length
    # c is arm length
    # y is neck length
    # a is head_arm_change
    # s is shoulder_length
    # Drawing a picture with the arm and neck as a right triangle is basically necessary to understand this

    arm_change = (math.sqrt((head_arm_change * total_length - neck_length) * (head_arm_change * total_length + neck_length)) / arm_length) - (shoulder_length / arm_length)

    return arm_change


def get_eye_height(obj):
    try:
        left_eye = get_bone("left_eye", obj)
        right_eye = get_bone("right_eye", obj)
    except KeyError as ke:
        raise RuntimeError(f'Cannot identify two eye bones: {ke}')

    eye_average = (left_eye.head + right_eye.head) / 2

    return eye_average[2]

def get_leg_length(arm):
    # Assumes exact symmetry between right and left legs
    return get_bone("left_leg", arm).head[2] - get_lowest_point()

def get_leg_proportions(arm):
    # Gets the relative lengths of each portion of the leg
    l = [
        (get_bone('left_leg', arm).head[2] + get_bone('right_leg', arm).head[2]) / 2,
        (get_bone('left_knee', arm).head[2] + get_bone('right_knee', arm).head[2]) / 2,
        (get_bone('left_ankle', arm).head[2] + get_bone('right_ankle', arm).head[2]) / 2,
        get_lowest_point()
    ]

    total = l[0] - l[3]
    nl = list([1 - (i-l[3])/total for i in l])
    return nl, total


def scale_legs(arm, leg_scale_ratio, leg_thickness, scale_foot, thigh_percentage):

    leg_points, total_length = get_leg_proportions(arm)

    starting_portions = list([leg_points[i+1]-leg_points[i] for i in range(3)])
    print("starting_portions: {}".format(starting_portions))

    # Foot scale is the percentage of the final it'll take up.
    foot_portion = ((1 - leg_points[2]) * leg_thickness / leg_scale_ratio)
    if scale_foot:
        foot_portion = (1 - leg_points[2]) * leg_thickness
    print("Foot portion: {}".format(foot_portion))
    print("Leg thickness: {}, leg_scale_ratio: {}, leg_points: {}".format(leg_thickness, leg_scale_ratio, leg_points))

    leg_portion = 1 - foot_portion

    # TODO: Add switch for maintaining existing thigh/calf proportions, make default(?)
    thigh_portion = leg_portion * thigh_percentage
    calf_portion = leg_portion - thigh_portion

    print("calculated desired leg portions: {}".format([thigh_portion, calf_portion, foot_portion]))

    final_thigh_scale = (thigh_portion / starting_portions[0]) * leg_scale_ratio
    final_calf_scale = (calf_portion / starting_portions[1]) * leg_scale_ratio
    final_foot_scale = (foot_portion / starting_portions[2]) * leg_scale_ratio

    # Disable scaling from parent for bones
    scale_bones = ["left_knee", "right_knee", "left_ankle", "right_ankle"]
    saved_bone_inherit_scales = {}
    for b in scale_bones:
        bone = get_bone(b, arm)
        saved_bone_inherit_scales[b] = arm.data.bones[bone.name].inherit_scale

        arm.data.bones[bone.name].inherit_scale = "NONE"

    print(final_foot_scale)
    print("Calculated final scales: thigh {} calf {} foot {}".format(final_thigh_scale, final_calf_scale, final_foot_scale))

    for leg in [get_bone("left_leg", arm), get_bone("right_leg", arm)]:
        leg.scale = (leg_thickness, final_thigh_scale, leg_thickness)
    for knee in [get_bone("left_knee", arm), get_bone("right_knee", arm)]:
        knee.scale = (leg_thickness, final_calf_scale, leg_thickness)
    for foot in [get_bone("left_ankle", arm), get_bone("right_ankle", arm)]:
        foot.scale = (final_foot_scale, final_foot_scale, final_foot_scale)

    result_final_points, result_total_legs = get_leg_proportions(arm)
    print("Implemented leg portions: {}".format(result_final_points))
    # restore saved bone scaling states
    # for b in scale_bones:
    #     arm.data.bones[b].inherit_scale = saved_bone_inherit_scales[b]


def start_pose_mode_with_reset(arm):
    """Replacement for Cats 'start pose mode' operator"""
    # Ensure armature isn't hidden
    # FIXME: unhide_obj is broken. It looks like bpy.ops.object.mode_set doesn't even care about hide_set and only cares
    #  about hide_viewport not being True?
    # unhide_obj(arm)
    arm.hide_set(False)
    # Clear the current pose of the armature
    reset_current_pose(arm.pose.bones)
    # Ensure that the armature data is set to pose position, otherwise setting a pose has no effect
    arm.data.pose_position = 'POSE'
    # Exit to OBJECT mode with whatever is the currently active object
    bpy.ops.object.mode_set(mode='OBJECT')
    # Set the armature as the active object
    bpy.context.view_layer.objects.active = arm
    # Open the armature in pose mode
    bpy.ops.object.mode_set(mode='POSE')


def scale_to_floor(arm_to_legs, arm_thickness, leg_thickness, extra_leg_length, scale_hand, thigh_percentage, custom_scale_ratio):
    arm = get_armature()

    view_y = get_view_y(arm, custom_scale_ratio) + extra_leg_length
    eye_y = get_eye_height(arm)

    # TODO: add an option for people who *want* their legs below the floor.
    #
    # weirdos
    rescale_ratio = eye_y / view_y
    leg_height_portion = get_leg_length(arm) / eye_y

    # Enforces: rescale_leg_ratio * rescale_arm_ratio = rescale_ratio
    rescale_leg_ratio = rescale_ratio ** arm_to_legs
    rescale_arm_ratio = rescale_ratio ** (1-arm_to_legs)

    leg_scale_ratio = 1 - (1 - (1/rescale_leg_ratio)) / leg_height_portion
    arm_scale_ratio = calculate_arm_rescaling(arm, rescale_arm_ratio)

    print("Total required scale factor is %f" % rescale_ratio)
    print("Scaling legs by a factor of %f to %f" % (leg_scale_ratio, leg_scale_ratio * get_leg_length(arm)))
    print("Scaling arms by a factor of %f" % arm_scale_ratio)

    start_pose_mode_with_reset(arm)

    leg_thickness = leg_thickness + leg_scale_ratio * (1 - leg_thickness)
    arm_thickness = arm_thickness + arm_scale_ratio * arm_thickness


    scale_foot = False
    scale_legs(arm, leg_scale_ratio, leg_thickness, scale_foot, thigh_percentage)

    # This kept getting me - make sure arms are set to inherit scale
    for b in ["left_elbow", "right_elbow", "left_wrist", "right_wrist"]:
        get_bone(b, arm).bone.inherit_scale = "FULL"

    for armbone in [get_bone("left_arm", arm), get_bone("right_arm", arm)]:
        armbone.scale = (arm_thickness, arm_scale_ratio, arm_thickness)

    if not scale_hand:
        for hand in [get_bone("left_wrist", arm), get_bone("right_wrist", arm)]:
            hand.scale = (1 / arm_thickness, 1 / arm_scale_ratio, 1 / arm_thickness)

            result_final_points, result_total_legs = get_leg_proportions(arm)
        print("Implemented leg portions: {}".format(result_final_points))

    # Apply the pose as rest pose, updating the meshes and their shape keys if they have them
    apply_pose_to_rest()


def move_to_floor():

    arm = get_armature()
    unhide_obj(arm)
    dz = get_lowest_point()

    aloc = get_armature().location
    newOrigin = (aloc[0], aloc[1], dz)

    print("New origin point: {}".format(newOrigin))
    print("Moving origin down by %f"%dz)
    #print("Highest point is %f"%hp)

    meshes = get_body_meshes()
    for obj in meshes:
        hidden = obj.hide_get()
        obj.hide_set(False)

        bpy.context.view_layer.objects.active = obj
        obj.select_set(True)
        bpy.ops.object.mode_set(mode='OBJECT', toggle = False)
        bpy.context.scene.cursor.location = newOrigin
        bpy.ops.object.origin_set(type='ORIGIN_CURSOR')

        # This actually does the moving of the body
        obj.location = (aloc[0],aloc[1],0)
        obj.hide_set(hidden)
        obj.select_set(False)

    bpy.context.view_layer.objects.active = arm
    arm.select_set(True)
    bpy.ops.object.mode_set(mode='EDIT', toggle = False)
    before_zs = {b.name: (b.head.z, b.tail.z) for b in arm.data.edit_bones}
    use_x_mirror = arm.data.use_mirror_x
    # Moving bones in edit mode automatically moves symmetrical bones (by name) if use_mirror_x is enabled, but the
    # bones might be asymmetrical by design, so temporarily disable mirroring if it's enabled.
    arm.data.use_mirror_x = False
    for bone in arm.data.edit_bones:
        #bone.transform(mathutils.Matrix.Translation((0, 0, -dz)))
        bone.head.z = before_zs[bone.name][0] - dz
        bone.tail.z = before_zs[bone.name][1] - dz
        # for b in arm.data.edit_bones:
        #     if b.name != bone.name and b.head.z != before_zs[b.name]:

        #         print("ERROR: Bone %s also changed bone %s: %f to %f"%(bone.name, b.name, before_zs[b.name], b.head.z))
        #print("%s: %f -> %f: %f"%(bone.name, bz, az, bz - az))
    arm.data.use_mirror_x = use_x_mirror
    bpy.ops.object.mode_set(mode='EDIT', toggle = True)

    bpy.context.scene.cursor.location = (aloc[0],aloc[1],0)
    bpy.ops.object.origin_set(type='ORIGIN_CURSOR')
    arm.select_set(False)

def recursive_object_mode(obj):
    bpy.context.view_layer.objects.active = obj
    hidden = obj.hide_get()
    obj.hide_set(False)
    bpy.ops.object.mode_set(mode='OBJECT', toggle = False)
    for c in obj.children:
        if not obj_in_scene(c):
            continue
        if len(c.users_scene) == 0:
            continue
        if 'scale' in dir(c):
            recursive_object_mode(c)
    obj.hide_set(hidden)

def recursive_scale(obj):
    bpy.context.scene.cursor.location = obj.location
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    print("Scaling {} by {}".format(obj.name, 1 / obj.scale[0]))
    # Would fail on multi-user data, but is only called after applying as rest pose, which will replace multi-user data
    # with copies
    bpy.ops.object.transform_apply(scale = True, location = False, rotation = False, properties = False)

    for c in obj.children:
        if not obj_in_scene(c):
            continue
        if len(c.users_scene) == 0:
            continue
        if 'scale' in dir(c):
            recursive_scale(c)


def scale_to_height(new_height, scale_eyes):
    obj = get_armature()
    unhide_obj(obj)
    old_height = get_highest_point() - get_lowest_point()
    if scale_eyes:
        old_height = get_eye_height(obj) - get_lowest_point()

    print("Old height is %f"%old_height)

    scale_ratio = new_height / old_height
    print("Scaling by %f to achieve target height" % scale_ratio)
    bpy.context.scene.cursor.location = obj.location
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)

    obj.scale = obj.scale * scale_ratio

    recursive_object_mode(obj)
    recursive_scale(obj)

    rehide_obj(obj)

def center_model():
    arm = get_armature()
    arm.location = (0,0,0)


def rescale_main(new_height, arm_to_legs, arm_thickness, leg_thickness, extra_leg_length, scale_hand, thigh_percentage, custom_scale_ratio, scale_eyes):
    s = bpy.context.scene


    if not s.debug_no_adjust:
        scale_to_floor(arm_to_legs, arm_thickness, leg_thickness, extra_leg_length, scale_hand, thigh_percentage, custom_scale_ratio)
    if not s.debug_no_floor:
        move_to_floor()

    result_final_points, result_total_legs = get_leg_proportions(get_armature())
    print("Final Implemented leg portions: {}".format(result_final_points))

    if not s.debug_no_scale:
        scale_to_height(new_height, scale_eyes)

    if s.center_model:
        center_model()

    bpy.ops.object.select_all(action='DESELECT')

def point_bone(bone, point, spread_factor):
    v1 = (bone.tail - bone.head).normalized()
    v2 = (bone.head - point).normalized()

    # Need to transform the global rotation between the two vectors
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

    newbm = bone.matrix.to_quaternion()

    # Run the actual rotation twice to give us more range on the
    # rotation. The slerp should exactly remove one of these by
    # default, basically letting us extrapolate a bit.
    newbm.rotate(rotation_quat_pose)
    newbm.rotate(rotation_quat_pose)

    newbm.rotate(bone.matrix.inverted())

    oldbm = bone.matrix.to_quaternion()
    oldbm.rotate(bone.matrix.inverted())

    finalbm = oldbm.slerp(newbm, spread_factor / 2)

    bone.rotation_quaternion = finalbm

def spread_fingers(spare_thumb, spread_factor):
    obj = get_armature()
    start_pose_mode_with_reset(obj)
    for hand in [get_bone("right_wrist", obj), get_bone("left_wrist", obj)]:
        for finger in hand.children:
            if "thumb" in finger.name.lower() and spare_thumb:
                continue
            point_bone(finger, hand.head, spread_factor)
    apply_pose_to_rest()
    bpy.ops.object.mode_set(mode='OBJECT')
    bpy.ops.object.select_all(action='DESELECT')

def shrink_hips():
    arm = get_armature()

    bpy.context.view_layer.objects.active = arm
    arm.select_set(True)
    bpy.ops.object.mode_set(mode='EDIT', toggle = False)

    left_leg_name = get_bone('left_leg', arm).name
    right_leg_name = get_bone('right_leg', arm).name
    leg_start = (arm.data.edit_bones[left_leg_name].head[2] +arm.data.edit_bones[right_leg_name].head[2]) / 2
    spine_start = arm.data.edit_bones['Spine'].head[2]

    # Make the hip tiny - 90 of the way between the start of the legs
    # and the start of the spine

    arm.data.edit_bones['Hips'].head[2] = leg_start + (spine_start - leg_start) * .9
    arm.data.edit_bones['Hips'].head[1] = arm.data.edit_bones['Spine'].head[1]
    arm.data.edit_bones['Hips'].head[0] = arm.data.edit_bones['Spine'].head[0]

    bpy.ops.object.mode_set(mode='EDIT', toggle = True)
    bpy.ops.object.select_all(action='DESELECT')


_ZERO_ROTATION_QUATERNION = np.array([1, 0, 0, 0], dtype=np.single)


def reset_current_pose(pose_bones):
    """Resets the location, scale and rotation of each pose bone to the rest pose."""
    num_bones = len(pose_bones)
    # 3 components: X, Y, Z, set each bone to (0,0,0)
    pose_bones.foreach_set('location', np.zeros(num_bones * 3, dtype=np.single))
    # 3 components: X, Y, Z, set each bone to (1,1,1)
    pose_bones.foreach_set('scale', np.ones(num_bones * 3, dtype=np.single))
    # 4 components: W, X, Y, Z, set each bone to (1, 0, 0, 0)
    pose_bones.foreach_set('rotation_quaternion', np.tile(_ZERO_ROTATION_QUATERNION, num_bones))


def _create_armature_mod_for_apply(armature_obj, mesh_obj, preserve_volume):
    armature_mod = cast(bpy.types.ArmatureModifier, mesh_obj.modifiers.new('IMScalePoseToRest', 'ARMATURE'))
    armature_mod.object = armature_obj
    # Rotating joints tends to scale down neighboring geometry, up to nearly zero at 180 degrees from rest position. By
    # enabling Preserve Volume, this no longer happens, but there is a 'gap', a discontinuity when going past 180
    # degrees (presumably the rotation effectively jumps to negative when going past 180 degrees)
    # This does have an effect when scaling bones, but it's unclear if it's a beneficial effect or even noticeable in
    # most cases.
    armature_mod.use_deform_preserve_volume = preserve_volume
    return armature_mod


def _apply_armature_to_mesh_with_no_shape_keys(armature_obj, mesh_obj, preserve_volume):
    armature_mod = _create_armature_mod_for_apply(armature_obj, mesh_obj, preserve_volume)
    me = mesh_obj.data
    if me.users > 1:
        # Can't apply modifiers to multi-user data, so make a copy of the mesh and set it as the object's data
        me = me.copy()
        mesh_obj.data = me
    # In the unlikely case that there was already a modifier with the same name as the new modifier, the new
    # modifier will have ended up with a different name
    mod_name = armature_mod.name
    # Context override to let us run the modifier operators on mesh_obj, even if it's not the active object
    context_override = {'object': mesh_obj}
    # Moving the modifier to the first index will prevent an Info message about the applied modifier not being
    # first and potentially having unexpected results.
    if bpy.app.version >= (2, 90, 0):
        # modifier_move_to_index was added in Blender 2.90
        op_override(bpy.ops.object.modifier_move_to_index, context_override, modifier=mod_name, index=0)
    else:
        # The newly created modifier will be at the bottom of the list
        armature_mod_index = len(mesh_obj.modifiers) - 1
        # Move the modifier up until it's at the top of the list
        for _ in range(armature_mod_index):
            op_override(bpy.ops.object.modifier_move_up, context_override, modifier=mod_name)
    op_override(bpy.ops.object.modifier_apply, context_override, modifier=mod_name)


def _apply_armature_to_mesh_with_shape_keys(armature_obj, mesh_obj, preserve_volume):
    # The active shape key will be changed, so save the current active index, so it can be restored afterwards
    old_active_shape_key_index = mesh_obj.active_shape_key_index

    # Shape key pinning shows the active shape key in the viewport without blending; effectively what you see when
    # in edit mode. Combined with an armature modifier, we can use this to figure out the correct positions for all
    # the shape keys.
    # Save the current value, so it can be restored afterwards.
    old_show_only_shape_key = mesh_obj.show_only_shape_key
    mesh_obj.show_only_shape_key = True

    # Temporarily remove vertex_groups from and disable mutes on shape keys because they affect pinned shape keys
    me = mesh_obj.data
    if me.users > 1:
        # Imagine two objects in different places with the same mesh data. Both objects can move different amounts
        # (they can even have completely different vertex groups), but we can only apply the movement to one of these
        # objects, so create a copy and set that copy as mesh_obj's data.
        me = me.copy()
        mesh_obj.data = me
    shape_key_vertex_groups = []
    shape_key_mutes = []
    key_blocks = me.shape_keys.key_blocks
    for shape_key in key_blocks:
        shape_key_vertex_groups.append(shape_key.vertex_group)
        shape_key.vertex_group = ''
        shape_key_mutes.append(shape_key.mute)
        shape_key.mute = False

    # Temporarily disable all modifiers from showing in the viewport so that they have no effect
    mods_to_reenable_viewport = []
    for mod in mesh_obj.modifiers:
        if mod.show_viewport:
            mod.show_viewport = False
            mods_to_reenable_viewport.append(mod)

    # Temporarily add a new armature modifier
    armature_mod = _create_armature_mod_for_apply(armature_obj, mesh_obj, preserve_volume)

    # cos are xyz positions and get flattened when using the foreach_set/foreach_get functions, so the array length
    # will be 3 times the number of vertices
    co_length = len(me.vertices) * 3
    # We can re-use the same array over and over
    eval_verts_cos_array = np.empty(co_length, dtype=np.single)

    # The first shape key will be the first one we'll affect, so set it as active before we get the depsgraph to avoid
    # having to update the depsgraph
    mesh_obj.active_shape_key_index = 0
    # depsgraph lets us evaluate objects and get their state after the effect of modifiers and shape keys
    # Get the depsgraph
    depsgraph = bpy.context.evaluated_depsgraph_get()
    # Evaluate the mesh
    evaluated_mesh_obj = mesh_obj.evaluated_get(depsgraph)

    # The cos of the vertices of the evaluated mesh include the effect of the pinned shape key and all the
    # modifiers (in this case, only the armature modifier we added since all the other modifiers are disabled in
    # the viewport).
    # This combination gives the same effect as if we'd applied the armature modifier to a mesh with the same
    # shape as the active shape key, so we can simply set the shape key to the evaluated mesh position.
    #
    # Get the evaluated cos
    evaluated_mesh_obj.data.vertices.foreach_get('co', eval_verts_cos_array)
    # Set the 'basis' (reference) shape key
    key_blocks[0].data.foreach_set('co', eval_verts_cos_array)
    # And also set the mesh vertices to ensure that the two remain in sync
    me.vertices.foreach_set('co', eval_verts_cos_array)

    # For the remainder of the shape keys, we only need to update the shape key itself
    for i, shape_key in enumerate(key_blocks[1:], start=1):
        # As shape key pinning is enabled, when we change the active shape key, it will change the state of the mesh
        mesh_obj.active_shape_key_index = i

        # In order for the change to the active shape key to take effect, the depsgraph has to be updated
        depsgraph.update()

        # Get the cos of the vertices from the evaluated mesh
        evaluated_mesh_obj.data.vertices.foreach_get('co', eval_verts_cos_array)
        # And set the shape key to those same cos
        shape_key.data.foreach_set('co', eval_verts_cos_array)

    # Restore temporarily changed attributes and remove the added armature modifier
    for mod in mods_to_reenable_viewport:
        mod.show_viewport = True
    mesh_obj.modifiers.remove(armature_mod)
    for shape_key, vertex_group, mute in zip(me.shape_keys.key_blocks, shape_key_vertex_groups, shape_key_mutes):
        shape_key.vertex_group = vertex_group
        shape_key.mute = mute
    mesh_obj.active_shape_key_index = old_active_shape_key_index
    mesh_obj.show_only_shape_key = old_show_only_shape_key


def apply_pose_to_rest(preserve_volume=False):
    """Apply pose to armature and meshes, taking into account shape keys on the meshes.
    The armature must be in Pose mode."""
    arm = get_armature()
    meshes = get_body_meshes()
    for mesh_obj in meshes:
        me = cast(bpy.types.Mesh, mesh_obj.data)
        if me:
            if me.shape_keys and me.shape_keys.key_blocks:
                # The mesh has shape keys
                shape_keys = me.shape_keys
                key_blocks = shape_keys.key_blocks
                if len(key_blocks) == 1:
                    # The mesh only has a basis shape key, so we can remove it and then add it back afterwards
                    # Get basis shape key
                    basis_shape_key = key_blocks[0]
                    # Save the name of the basis shape key
                    original_basis_name = basis_shape_key.name
                    # Remove the basis shape key so there are now no shape keys
                    mesh_obj.shape_key_remove(basis_shape_key)
                    # Apply the pose to the mesh
                    _apply_armature_to_mesh_with_no_shape_keys(arm, mesh_obj, preserve_volume)
                    # Add the basis shape key back with the same name as before
                    mesh_obj.shape_key_add(name=original_basis_name)
                else:
                    # Apply the pose to the mesh, taking into account the shape keys
                    _apply_armature_to_mesh_with_shape_keys(arm, mesh_obj, preserve_volume)
            else:
                # The mesh doesn't have shape keys, so we can easily apply the pose to the mesh
                _apply_armature_to_mesh_with_no_shape_keys(arm, mesh_obj, preserve_volume)
    # Once the mesh and shape keys (if any) have been applied, the last step is to apply the current pose of the
    # bones as the new rest pose.
    #
    # From the poll function, armature_obj must already be in pose mode, but it's possible it might not be the
    # active object e.g., the user has multiple armatures opened in pose mode, but a different armature is currently
    # active. We can use an operator override to tell the operator to treat armature_obj as if it's the active
    # object even if it's not, skipping the need to actually set armature_obj as the active object.
    bpy.ops.pose.armature_apply({'active_object': arm})


class ArmatureOperator(bpy.types.Operator):
    # poll_message_set was added in 3.0
    if not hasattr(bpy.types.Operator, 'poll_message_set'):
        @classmethod
        def poll_message_set(cls, message, *args):
            pass

    @classmethod
    def poll(cls, context: bpy.types.Context) -> bool:
        if get_armature() is None:
            cls.poll_message_set("Armature not found. Select an armature as active or ensure an armature is set in Cats"
                                 " if you have Cats installed.")
            return False
        return True


class ArmatureRescale(ArmatureOperator):
    """Script to scale most aspects of an armature for use in vrchat"""
    bl_idname = "armature.rescale"
    bl_label = "Rescale Armature"
    bl_options = {'REGISTER', 'UNDO'}

    #set_properties()
    # target_height: bpy.types.Scene.target_height
    # arm_to_legs: bpy.types.Scene.arm_to_legs
    # arm_thickness: bpy.types.Scene.arm_thickness
    # leg_thickness: bpy.types.Scene.leg_thickness
    # extra_leg_length: bpy.types.Scene.extra_leg_length
    # scale_hand: bpy.types.Scene.scale_hand
    # thigh_percentage: bpy.types.Scene.thigh_percentage
    # scale_eyes: bpy.types.Scene.scale_eyes

    def execute(self, context):
        rescale_main(
            self.target_height,
            self.arm_to_legs / 100.0,
            self.arm_thickness / 100.0,
            self.leg_thickness / 100.0,
            self.extra_leg_length,
            self.scale_hand,
            self.thigh_percentage / 100.0,
            self.custom_scale_ratio,
            self.scale_eyes,
        )
        return {'FINISHED'}

    def invoke(self, context, event):
        s = context.scene
        self.target_height = s.target_height
        self.arm_to_legs = s.arm_to_legs
        self.arm_thickness = s.arm_thickness
        self.leg_thickness = s.leg_thickness
        self.extra_leg_length = s.extra_leg_length
        self.scale_hand = s.scale_hand
        self.thigh_percentage = s.thigh_percentage
        self.custom_scale_ratio = s.custom_scale_ratio
        self.scale_eyes = s.scale_eyes


        return self.execute(context)


class ArmatureSpreadFingers(ArmatureOperator):
    """Spreads the fingers on a humanoid avatar"""
    bl_idname = "armature.spreadfingers"
    bl_label = "Spread Fingers"
    bl_options = {'REGISTER', 'UNDO'}

    # spare_thumb: bpy.types.Scene.spare_thumb
    # spread_factor: bpy.types.Scene.spread_factor

    def execute(self, context):
        spread_fingers(self.spare_thumb, self.spread_factor)
        return {'FINISHED'}

    def invoke(self, context, event):
        s = context.scene
        self.spare_thumb = s.spare_thumb
        self.spread_factor = s.spread_factor

        return self.execute(context)

class ArmatureShrinkHip(ArmatureOperator):
    """Shrinks the hip bone in a humaniod avatar to be much closer to the spine location"""
    bl_idname = "armature.shrink_hips"
    bl_label = "Shrink Hips"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        shrink_hips()
        return {'FINISHED'}

class UIGetCurrentHeight(ArmatureOperator):
    """Sets target height based on the current height"""
    bl_idname = "armature.get_avatar_height"
    bl_label = "Get Current Avatar Height"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        height = 1.5 # Placeholder
        if context.scene.scale_eyes:
            height = get_eye_height(get_armature())
        else:
            height = get_highest_point()
        context.scene.target_height = height
        return {'FINISHED'}


class UIGetScaleRatio(ArmatureOperator):
    """Gets the custom scaling ratio based on the current avatar's proportions"""
    bl_idname = "armature.get_scale_ratio"
    bl_label = "Get Current Avatar Scale Ratio"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        scale = get_current_scaling(get_armature())
        context.scene.custom_scale_ratio = scale
        return {'FINISHED'}


def ops_register():
    print("Registering Armature tuning add-on")
    bpy.utils.register_class(ArmatureRescale)

    bpy.utils.register_class(ArmatureSpreadFingers)

    bpy.utils.register_class(ArmatureShrinkHip)

    bpy.utils.register_class(UIGetCurrentHeight)

    bpy.utils.register_class(UIGetScaleRatio)

    print("Registering Armature tuning add-on")

def ops_unregister():
    print("Attempting to unregister armature turing add-on")
    bpy.utils.unregister_class(ArmatureRescale)
    bpy.utils.unregister_class(ArmatureSpreadFingers)
    bpy.utils.unregister_class(ArmatureShrinkHip)
    bpy.utils.unregister_class(UIGetCurrentHeight)
    bpy.utils.unregister_class(UIGetScaleRatio)
    print("Unregistering Armature tuning add-on")

if __name__ == "__main__":
    register()
