class RequestException(Exception):
    pass


class Response:
    def __init__(self, status_code=200, payload=None):
        self.status_code = status_code
        self._payload = payload or {}

    def json(self):
        return self._payload


def get(url, timeout=None):
    if "network/state" in url:
        return Response(payload={"feature1": 0.1, "feature2": 0.2, "feature3": 0.3})
    if "nfv/state" in url:
        return Response(payload={"vnf_count": 2, "cpu_utilization": 0.5})
    return Response(payload={})


def post(url, headers=None, data=None, timeout=None):
    return Response(status_code=200, payload={"status": "ok"})
