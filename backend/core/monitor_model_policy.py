def hidden_monitor_cameras(deployment, workflows, registered):
    controlled, visible = set(), set()
    registered = set(registered)
    if deployment:
        visible.update(registered if deployment.get("all_cameras") else set(deployment.get("camera_ids", [])) & registered)
    for workflow in workflows:
        nodes = workflow.get("definition", {}).get("nodes", [])
        cameras = {camera for node in nodes if node.get("type") == "source"
                   for camera in node.get("config", {}).get("camera_ids", [])} & registered
        controlled.update(cameras)
        outputs = [node for node in nodes if node.get("type") == "display"]
        if not outputs or any(node.get("config", {}).get("monitor", True) for node in outputs):
            visible.update(cameras)
    return frozenset(controlled - visible)


def apply_monitor_visibility(payload, hidden):
    if not hidden:
        return payload
    return dict(payload, streams=[dict(stream, objects=[], monitor_hidden=True)
                                 if stream.get("cam_id") in hidden else stream
                                 for stream in payload.get("streams", [])])


def filter_monitor_metadata(payload, accepts_stream):
    source = payload.get("source", "deepstream")
    if source in {"identity_template", "identity_people"}:
        return None
    streams = []
    for stream in payload.get("streams", []):
        camera_id = stream.get("cam_id")
        model_id = stream.get("model_id")
        if source == "custom_deepstream":
            if not accepts_stream(camera_id, model_id):
                continue
            objects = [obj for obj in stream.get("objects", [])
                       if isinstance(obj, dict) and obj.get("model_id") == model_id]
        else:
            objects = [dict(obj, mask=None, mask_stale=True, model_id=None, label=None, identity_registered=False)
                       for obj in stream.get("objects", []) if isinstance(obj, dict)]
        streams.append(dict(stream, objects=objects))
    return dict(payload, streams=streams) if streams else None
