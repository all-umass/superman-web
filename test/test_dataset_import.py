import unittest
from mock import Mock

from backend import MatplotlibServer
from backend.handlers.upload import DatasetUploadHandler


class TestDatasetImportRoutes(unittest.TestCase):
    def setUp(self):
        self.app = MatplotlibServer([], cookie_secret='foobar')

    def test_csv_spectra(self):
        req = Mock(cookies=dict())
        req.arguments = dict(ds_name=['test'], ds_kind=[
                             'Raman'], desc=['foobar'])
        req.files = dict(spectra=[dict(body=b'wave,foo,bar\n2,4,3\n3,4,4\n')])
        h = DatasetUploadHandler(self.app, req)
        h.write = Mock()
        h.finish = Mock()
        h.post()

        self.assertEqual(len(h.write.call_args_list), 1)
        self.assertEqual(len(h.finish.call_args_list), 0)
        args, kwargs = h.write.call_args
        self.assertEqual(args, (u'/explorer?ds_kind=Raman&ds_name=test',))
        self.assertEqual(kwargs, {})


if __name__ == '__main__':
    unittest.main()
