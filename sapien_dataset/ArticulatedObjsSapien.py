class ArticulatedObjectSapien():
    def __init__(self, class_id, geometry, parameters, xml, pose, rotation, handle_name, joint_name, act_idx):
        self.type = class_id
        self.geom = geometry
        self.params = parameters
        self.pose = pose
        self.rotation = rotation
        self.xml = xml
        self.handle_name = handle_name
        self.joint_name = joint_name
        self.act_idx = act_idx
        # self.cam_params = cam_params


class Microwave(ArticulatedObjectSapien):
    def __init__(self, class_id, geometry, parameters, xml, pose, rotation):
        super(Microwave, self).__init__(class_id, geometry, parameters, xml, pose, rotation, 'handle', 0)
        self.control = -0.2


class Dishwasher(ArticulatedObjectSapien):
    def __init__(self, class_id, geometry, parameters, xml, pose, rotation):
        super(Dishwasher, self).__init__(class_id, geometry, parameters, xml, pose, rotation, 'door', 0)
        self.control = 0.2
