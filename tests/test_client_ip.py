import unittest

from flask import Flask, request

import app
from app.routes.student import _safe_ip


def make_app(hops):
    web = Flask(__name__)
    web.config["TRUSTED_PROXY_HOPS"] = hops
    app._trust_proxies(web)

    @web.route("/ip")
    def ip():
        return {"remote_addr": request.remote_addr, "safe_ip": _safe_ip()}

    return web


def get_ip(web, forwarded_for):
    # cloudflared connects from 127.0.0.1 and appends the real client last.
    return web.test_client().get(
        "/ip", headers={"X-Forwarded-For": forwarded_for},
        environ_base={"REMOTE_ADDR": "127.0.0.1"},
    ).json


class ClientIpTests(unittest.TestCase):
    def test_one_trusted_hop_uses_the_address_the_proxy_appended(self):
        ip = get_ip(make_app(1), "6.6.6.6, 203.0.113.9")
        self.assertEqual(ip["remote_addr"], "203.0.113.9")

    def test_client_supplied_first_entry_is_not_used(self):
        # A client can send its own X-Forwarded-For; only the proxy's entry counts.
        ip = get_ip(make_app(1), "6.6.6.6, 203.0.113.9")
        self.assertEqual(ip["safe_ip"], "203.0.113.9")

    def test_no_trusted_hops_ignores_the_header(self):
        ip = get_ip(make_app(0), "203.0.113.9")
        self.assertEqual(ip, {"remote_addr": "127.0.0.1", "safe_ip": "127.0.0.1"})


if __name__ == "__main__":
    unittest.main()
