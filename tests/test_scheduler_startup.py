import unittest
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import Mock, patch

from flask import Flask
from app import scheduler


class SchedulerStartupTests(unittest.TestCase):
    def make_app(self, debug, reloader_child):
        web = Flask(__name__)
        web.debug = debug
        web.add_url_rule('/', view_func=lambda: 'ok')
        job = Mock(running=False)
        job.start.side_effect = lambda: setattr(job, 'running', True)
        with patch.dict('os.environ', {'WERKZEUG_RUN_MAIN': reloader_child}), \
                patch.object(scheduler, 'BackgroundScheduler', return_value=job):
            scheduler.start_scheduler(web)
        return web, job

    def test_serving_request_starts_once_in_every_mode(self):
        for debug, child in [(True, ''), (True, 'true'), (False, '')]:
            with self.subTest(debug=debug, child=child):
                web, job = self.make_app(debug, child)
                job.start.assert_not_called()
                with web.test_client() as client:
                    self.assertEqual(client.get('/').status_code, 200)
                    self.assertEqual(client.get('/').status_code, 200)
                job.start.assert_called_once()
                job.get_job.assert_called_once_with('session_manager')
                job.get_job.return_value.modify.assert_called_once()
                self.assertIsNotNone(job.get_job.return_value.modify.call_args.kwargs['next_run_time'])

    def test_reloader_supervisor_does_not_start_jobs(self):
        _, job = self.make_app(True, '')
        job.start.assert_not_called()

    def test_concurrent_first_requests_start_only_once(self):
        web, job = self.make_app(True, '')
        def request(_):
            with web.test_client() as client:
                return client.get('/').status_code
        with ThreadPoolExecutor(max_workers=8) as pool:
            self.assertEqual(list(pool.map(request, range(16))), [200] * 16)
        job.start.assert_called_once()


if __name__ == '__main__':
    unittest.main()
