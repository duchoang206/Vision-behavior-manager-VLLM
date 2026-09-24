import copy
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import MagicMock

from fastapi import FastAPI
from fastapi.testclient import TestClient

from core.dashboard_auth import require_dashboard_user
from core.metadata_fusion import MetadataFusion
from core.model_contract import infer_config, inspect_onnx, parse_labels
from core.workflow_definition import validate_definition
from core.workflow_runtime import GraphEvaluator, WorkflowRuntime
from routers.models import create_model_router


class ModelContractTests(unittest.TestCase):
    def setUp(self):
        import onnx
        self.onnx = onnx
        self.folder = tempfile.TemporaryDirectory()
        self.path = Path(self.folder.name) / 'best.onnx'
        tensor = onnx.helper.make_tensor('values', onnx.TensorProto.FLOAT, [1, 6, 8400], [0.] * 50400)
        graph = onnx.helper.make_graph([onnx.helper.make_node('Constant', [], ['output0'], value=tensor)], 'detector',
            [onnx.helper.make_tensor_value_info('images', onnx.TensorProto.FLOAT, [1, 3, 640, 640])],
            [onnx.helper.make_tensor_value_info('output0', onnx.TensorProto.FLOAT, [1, 6, 8400])])
        self.model = onnx.helper.make_model(graph, opset_imports=[onnx.helper.make_opsetid('', 17)])
        onnx.helper.set_model_props(self.model, {'task': 'detect', 'names': "{0: 'person', 1: 'helmet'}"})

    def tearDown(self):
        self.folder.cleanup()

    def inspect(self):
        self.onnx.save(self.model, str(self.path))
        return inspect_onnx(self.path)

    def test_metadata_names_and_order(self):
        result = self.inspect()
        self.assertEqual(result['labels'], ['person', 'helmet'])
        self.assertEqual(result['shape'], [1, 3, 640, 640])
        self.assertEqual(len(result['sha256']), 64)
        self.assertEqual(parse_labels('{"1": "robot", "0": "person"}'), ['person', 'robot'])

    def test_missing_names_require_manual_mapping(self):
        self.onnx.helper.set_model_props(self.model, {'task': 'detect'})
        with self.assertRaises(ValueError):
            self.inspect()
        self.assertEqual(inspect_onnx(self.path, ['robot', 'rack'])['labels_source'], 'manual')

    def test_pose_dynamic_wrong_class_count_rejected(self):
        original = copy.deepcopy(self.model)
        self.onnx.helper.set_model_props(self.model, {'task': 'pose', 'names': "{0: 'person'}"})
        with self.assertRaises(ValueError):
            self.inspect()
        self.model = copy.deepcopy(original)
        self.model.graph.input[0].type.tensor_type.shape.dim[0].dim_value = 2
        with self.assertRaises(ValueError):
            self.inspect()
        self.model = original
        self.onnx.helper.set_model_props(self.model, {'task': 'detect', 'names': "{0: 'person'}"})
        with self.assertRaises(ValueError):
            self.inspect()

    def test_pose_contract_generates_native_deepstream_config(self):
        tensor = self.onnx.helper.make_tensor('values', self.onnx.TensorProto.FLOAT, [1, 56, 8400], [0.] * (56 * 8400))
        graph = self.onnx.helper.make_graph([self.onnx.helper.make_node('Constant', [], ['output0'], value=tensor)], 'pose',
            [self.onnx.helper.make_tensor_value_info('images', self.onnx.TensorProto.FLOAT, [1, 3, 640, 640])],
            [self.onnx.helper.make_tensor_value_info('output0', self.onnx.TensorProto.FLOAT, [1, 56, 8400])])
        model = self.onnx.helper.make_model(graph, opset_imports=[self.onnx.helper.make_opsetid('', 17)])
        self.onnx.helper.set_model_props(model, {'task': 'pose', 'names': "{0: 'person'}"})
        self.onnx.save(model, str(self.path))
        metadata = inspect_onnx(self.path)
        config = infer_config('/models/pose', metadata, '/opt/visionmanager/libperson_pose_parser.so')
        self.assertEqual('pose', metadata['model_type'])
        self.assertIn('parse-bbox-func-name=NvDsInferParseYoloV8Pose', config)
        self.assertIn('output-tensor-meta=1', config)

    def test_external_weights_rejected_without_reading(self):
        tensor = self.model.graph.node[0].attribute[0].t
        tensor.data_location = self.onnx.TensorProto.EXTERNAL
        entry = tensor.external_data.add()
        entry.key, entry.value = 'location', '/etc/passwd'
        self.path.write_bytes(self.model.SerializeToString())
        with self.assertRaisesRegex(ValueError, 'self-contained'):
            inspect_onnx(self.path)

    def test_invalid_and_executable_labels_rejected(self):
        for value in ['__import__("os").system("false")', '{2:"robot"}', ['Robot', 'robot'], ['person\nrobot'], ['a;b']]:
            with self.subTest(value=value), self.assertRaises(ValueError):
                parse_labels(value)


class CustomPipelineTests(unittest.TestCase):
    def setUp(self):
        self.model_id = 'a' * 32
        self.graph = {'name': 'custom', 'nodes': [
            {'id': 'source', 'type': 'source', 'config': {'camera_ids': ['cam']}},
            {'id': 'detect', 'type': 'detector', 'config': {'classes': ['helmet'], 'model_id': self.model_id}},
            {'id': 'display', 'type': 'display', 'config': {}}],
            'edges': [{'source': 'source', 'target': 'detect'}, {'source': 'detect', 'target': 'display'}]}
        self.model = {'id': self.model_id, 'state': 'ready', 'labels': ['helmet', 'person']}

    def test_deploy_requires_ready_and_matching_labels(self):
        self.assertFalse(validate_definition(self.graph, ['cam'])['valid'])
        self.assertTrue(validate_definition(self.graph, ['cam'], models=[self.model])['valid'])
        self.model['state'] = 'building'
        self.assertFalse(validate_definition(self.graph, ['cam'], models=[self.model])['valid'])
        self.model['state'] = 'ready'
        self.graph['nodes'][1]['config']['classes'] = ['robot']
        self.assertFalse(validate_definition(self.graph, ['cam'], models=[self.model])['valid'])

    def test_detector_filters_exact_model_not_unrelated_metadata(self):
        engine = GraphEvaluator(self.graph, ['source', 'detect', 'display'])
        objects = [{'id': 1, 'class': 'helmet', 'confidence': .9, 'model_id': self.model_id},
                   {'id': 2, 'class': 'helmet', 'confidence': .9, 'model_id': 'other'}]
        summary, actions = engine.evaluate('cam', objects, 100)
        self.assertEqual(summary['display']['count'], 1)
        self.assertEqual(summary['display']['objects'][0]['id'], 1)

    def test_custom_and_mask_mailboxes_do_not_overwrite(self):
        runtime = WorkflowRuntime(MagicMock(), MagicMock())
        runtime.camera_ids = {'cam'}
        now = time.time() * 1000
        runtime.publish({'source': 'custom_deepstream', 'timestamp': now, 'streams': [{'cam_id': 'cam', 'model_id': self.model_id, 'objects': []}]})
        runtime.publish({'source': 'identity_template', 'timestamp': now + 1, 'streams': [{'cam_id': 'cam', 'objects': []}]})
        self.assertEqual(len(runtime.pending), 2)
        self.assertEqual(runtime.pending[('cam', self.model_id)][2], self.model_id)

    def test_fusion_keeps_masks_and_all_custom_classes_without_identity_guess(self):
        fusion = MetadataFusion()
        mask = {'polygons': [[[.1, .1], [.2, .1], [.2, .2]]]}
        fusion.update('cam', 'identity_template', [{'id': 1, 'class': 'robot', 'label': 'Robot_2001', 'mask': mask}], now=1)
        objects = fusion.update('cam', 'custom_deepstream', [{'id': 2, 'class': 'robot', 'model_id': self.model_id},
            {'id': 3, 'class': 'helmet', 'model_id': self.model_id}], now=1.1)
        self.assertEqual(len(objects), 2)
        self.assertEqual(objects[0]['id'], 2)
        self.assertIsNone(objects[0]['label'])
        self.assertEqual(len(fusion.update('cam', 'custom_deepstream', [], now=1.2)), 0)
        self.assertEqual(fusion.update('cam', 'identity_template', [], now=3), [])

    def test_upload_api_auth_size_and_offset(self):
        store = MagicMock(chunk_size=2 * 1024 * 1024, max_size=512 * 1024 * 1024)
        store.append.return_value = {'received_bytes': 3}
        app = FastAPI()
        app.include_router(create_model_router(store, lambda: {}))
        client = TestClient(app)
        self.assertEqual(client.post('/api/models', json={}).status_code, 401)
        app.dependency_overrides[require_dashboard_user] = lambda: 'admin'
        self.assertEqual(client.put('/api/models/abc/file?offset=-1', content=b'abc').status_code, 422)
        self.assertEqual(client.put('/api/models/abc/file?offset=0', content=b'x' * (2 * 1024**2 + 1)).status_code, 413)
        self.assertEqual(client.put('/api/models/abc/file?offset=0', content=b'abc').status_code, 200)
        store.append.assert_called_once_with('abc', 0, bytearray(b'abc'))


if __name__ == '__main__':
    unittest.main()
