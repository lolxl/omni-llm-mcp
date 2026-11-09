"""Synthetic local-socket contract checks; no model processes or external services."""
import json
import os
from pathlib import Path
import socket
import tempfile
import threading
import unittest

from providers.session_router_unix import UnixAdapter, receive_once


class FakeRegistry:
    def __init__(self):
        self.record = None
        self.ready = threading.Event()

    def lookup(self, **kw):
        from providers.session_router import RouterError
        if self.record is None:
            raise RouterError('caller_not_attached')
        return self.record

    def set_route(self, **kw):
        self.record.update(adapter=kw['adapter'], route=kw['route'])
        self.ready.set()
        return self.record

    def attach(self, session_key, incarnation, cwd, **kw):
        self.record = dict(kw, presence_id='synthetic-recipient', incarnation=incarnation)
        self.ready.set()
        return self.record


class UnixTransportTests(unittest.TestCase):
    def start_receiver(self, timeout=.5):
        registry = FakeRegistry()
        result = []
        thread = threading.Thread(target=lambda: result.append(receive_once(
            registry, 'synthetic-session', 'inc-1', '/tmp', timeout_s=timeout)))
        thread.start()
        self.assertTrue(registry.ready.wait(1))
        self.addCleanup(lambda: thread.join(2))
        return registry, result, thread

    def test_unicode_multiline_handoff_and_no_body_in_registration(self):
        registry, result, thread = self.start_receiver()
        body = 'synthetic payload: 雪\nsecond line\n"quoted"'
        target = dict(registry.record, _sender_id='synthetic-sender')
        sent = UnixAdapter().guarded_forward(target, body, expected_incarnation='inc-1')
        thread.join(2)
        self.assertEqual(sent, {'ok': True, 'status': 'handoff'})
        self.assertEqual(result[0]['body'], body)
        self.assertEqual(result[0]['sender_id'], 'synthetic-sender')
        self.assertIn('untrusted', result[0]['notice'])
        self.assertNotIn(body, json.dumps(registry.record, ensure_ascii=False))
        self.assertFalse(Path(registry.record['route']['path']).exists())

    def test_wrong_nonce_or_incarnation_sends_no_payload(self):
        for changed in ('nonce', 'incarnation', 'presence_id'):
            with self.subTest(changed=changed), tempfile.TemporaryDirectory(dir=os.path.realpath('/tmp')) as directory:
                path = os.path.join(directory, 's')
                record = dict(presence_id='synthetic-recipient', incarnation='inc-1',
                              route={'path': path, 'nonce': 'a' * 64})
                server = socket.socket(socket.AF_UNIX)
                server.bind(path)
                os.chmod(path, 0o600)
                server.listen(1)
                server.settimeout(1)
                received = []
                def peer():
                    with server, server.accept()[0] as conn:
                        conn.settimeout(1)
                        hello = dict(protocol='omni-session-unix', version=1,
                                     presence_id=record['presence_id'], incarnation='inc-1', nonce='a'*64)
                        hello[changed] = 'changed'
                        conn.sendall(json.dumps(hello).encode() + b'\n')
                        received.append(conn.recv(4096))
                thread = threading.Thread(target=peer)
                thread.start()
                sent = UnixAdapter().guarded_forward(record, 'synthetic-secret-marker', expected_incarnation='inc-1')
                thread.join(2)
                self.assertEqual(sent, {'ok': False, 'reason': 'identity_changed'})
                self.assertEqual(received, [b''])

    def test_replacement_does_not_receive_payload(self):
        registry, result, thread = self.start_receiver()
        path = registry.record['route']['path']
        os.unlink(path)
        replacement = socket.socket(socket.AF_UNIX)
        replacement.bind(path)
        os.chmod(path, 0o600)
        replacement.listen(1)
        replacement.settimeout(1)
        received = []
        def wrong_peer():
            with replacement, replacement.accept()[0] as conn:
                conn.settimeout(1)
                hello = dict(protocol='omni-session-unix', version=1, presence_id='replacement',
                             incarnation='inc-1', nonce='b' * 64)
                conn.sendall(json.dumps(hello).encode() + b'\n')
                received.append(conn.recv(4096))
        other = threading.Thread(target=wrong_peer)
        other.start()
        sent = UnixAdapter().guarded_forward(registry.record, 'synthetic-marker', expected_incarnation='inc-1')
        other.join(2)
        thread.join(2)
        self.assertEqual(sent['reason'], 'identity_changed')
        self.assertEqual(received, [b''])
        self.assertTrue(os.path.exists(path), 'receiver must preserve replacement inode')
        self.assertEqual(result[0]['reason'], 'timeout')
        os.unlink(path)
        os.rmdir(os.path.dirname(path))

    def test_connected_socket_survives_path_replacement(self):
        with tempfile.TemporaryDirectory(dir=os.path.realpath('/tmp')) as directory:
            path = os.path.join(directory, 's')
            record = dict(presence_id='synthetic-recipient', incarnation='inc-1',
                          route={'path': path, 'nonce': 'a' * 64})
            listener = socket.socket(socket.AF_UNIX)
            listener.bind(path)
            os.chmod(path, 0o600)
            listener.listen(1)
            listener.settimeout(1)
            delivered = []
            replacement = socket.socket(socket.AF_UNIX)
            def peer():
                with listener, listener.accept()[0] as conn:
                    conn.settimeout(1)
                    os.unlink(path)
                    replacement.bind(path)
                    os.chmod(path, 0o600)
                    replacement.listen(1)
                    hello = dict(protocol='omni-session-unix', version=1,
                                 presence_id=record['presence_id'], incarnation='inc-1', nonce='a'*64)
                    conn.sendall(json.dumps(hello).encode() + b'\n')
                    data = b''
                    while not data.endswith(b'\n'):
                        data += conn.recv(4096)
                    delivered.append(json.loads(data)['body'])
                    conn.sendall(b'{"ok":true,"status":"handoff"}\n')
            thread = threading.Thread(target=peer)
            thread.start()
            try:
                sent = UnixAdapter().guarded_forward(record, 'synthetic-connected', expected_incarnation='inc-1')
                thread.join(2)
                self.assertEqual(sent, {'ok': True, 'status': 'handoff'})
                self.assertEqual(delivered, ['synthetic-connected'])
                replacement.settimeout(.05)
                with self.assertRaises(socket.timeout):
                    replacement.accept()
            finally:
                replacement.close()

    def test_timeout_and_stale_endpoint_are_explicit(self):
        registry, result, thread = self.start_receiver(timeout=.05)
        thread.join(2)
        self.assertEqual(result, [{'ok': False, 'reason': 'timeout'}])
        sent = UnixAdapter().guarded_forward(registry.record, 'synthetic-marker', expected_incarnation='inc-1')
        self.assertEqual(sent, {'ok': False, 'reason': 'endpoint_unavailable'})

    def test_symlink_directory_rejected(self):
        with tempfile.TemporaryDirectory(dir=os.path.realpath('/tmp')) as directory:
            link = os.path.join(directory, 'link')
            os.symlink(directory, link)
            self.assertEqual(receive_once(FakeRegistry(), 's', 'i', '/tmp', socket_dir=link),
                             {'ok': False, 'reason': 'invalid_route'})

    def test_malformed_and_disconnect_are_structured(self):
        for payload, reason in ((b'not json\n', 'protocol_error'),
                                (b'{"body":false}\n', 'invalid_message'),
                                (b'', 'protocol_error')):
            with self.subTest(reason=reason):
                registry, result, thread = self.start_receiver()
                with socket.socket(socket.AF_UNIX) as conn:
                    conn.connect(registry.record['route']['path'])
                    hello = b''
                    while not hello.endswith(b'\n'):
                        hello += conn.recv(4096)
                    if payload:
                        conn.sendall(payload)
                    conn.shutdown(socket.SHUT_WR)
                thread.join(2)
                self.assertEqual(result, [{'ok': False, 'reason': reason}])

    def test_successive_receives_refresh_route(self):
        registry, result, thread = self.start_receiver(timeout=.05)
        thread.join(2)
        old_route = registry.record['route'].copy()
        registry.ready.clear()
        second = []
        thread = threading.Thread(target=lambda: second.append(receive_once(
            registry, 'synthetic-session', 'inc-1', '/tmp', timeout_s=.5)))
        thread.start()
        self.assertTrue(registry.ready.wait(1))
        self.assertNotEqual(old_route, registry.record['route'])
        sent = UnixAdapter().guarded_forward(registry.record, 'synthetic-repeat', expected_incarnation='inc-1')
        thread.join(2)
        self.assertEqual(sent, {'ok': True, 'status': 'handoff'})
        self.assertEqual(second[0]['body'], 'synthetic-repeat')

    def test_shared_directory_rejected(self):
        with tempfile.TemporaryDirectory(dir=os.path.realpath('/tmp')) as directory:
            os.chmod(directory, 0o755)
            self.assertEqual(receive_once(FakeRegistry(), 's', 'i', '/tmp', socket_dir=directory),
                             {'ok': False, 'reason': 'invalid_route'})

    def test_invalid_body_is_rejected_before_connect(self):
        self.assertEqual(UnixAdapter().guarded_forward({}, 'x' * 65537, expected_incarnation='i'),
                         {'ok': False, 'reason': 'invalid_message'})


if __name__ == '__main__':
    unittest.main()
