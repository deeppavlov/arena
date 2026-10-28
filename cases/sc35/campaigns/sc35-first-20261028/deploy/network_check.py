import json
import os
import subprocess
import sys

PROBE = '''
import json, socket, urllib.request
checks = json.loads(__import__("os").environ["CHECKS"])
results = []
for host, port, expected in checks:
    try:
        with socket.create_connection((host, port), timeout=2):
            actual = True
    except OSError:
        actual = False
    results.append({"host": host, "port": port, "expected": expected, "actual": actual})
print(json.dumps(results))
raise SystemExit(any(row["actual"] != row["expected"] for row in results))
'''


def main():
    image = os.environ["ARENA_EVALUATOR_IMAGE"]
    networks = {
        "sc35-first-researchers": [
            ("train-evaluator", 8080, True),
            ("public-evaluator", 8080, True),
            ("private-evaluator", 8080, True),
            ("shared-tracking", 5000, True),
            ("researcher-gateway", 8081, True),
            ("internet-proxy", 3128, True),
            ("172.31.91.1", 22, False),
            ("172.31.91.1", 5050, False),
            ("172.31.95.2", 5000, False),
            ("172.31.95.3", 5000, False),
            ("172.28.0.2", 5000, False),
            ("172.31.92.10", 8081, False),
            ("172.31.81.10", 8080, False),
            ("172.31.81.20", 5000, False),
            ("172.31.85.3", 5000, False),
            ("172.31.71.10", 8080, False),
            ("172.31.71.20", 5000, False),
            ("172.31.75.3", 5000, False),
            ("1.1.1.1", 443, False),
        ],
        "sc35-first-workers": [
            ("model-gateway", 8081, True),
            ("172.31.91.10", 8080, False),
            ("172.31.91.40", 3128, False),
            ("172.31.92.1", 22, False),
            ("172.31.92.1", 5050, False),
            ("172.31.95.2", 5000, False),
            ("172.31.95.3", 5000, False),
            ("172.28.0.2", 5000, False),
            ("172.31.81.10", 8080, False),
            ("172.31.81.20", 5000, False),
            ("172.31.85.3", 5000, False),
            ("172.31.71.10", 8080, False),
            ("172.31.71.20", 5000, False),
            ("172.31.75.3", 5000, False),
            ("1.1.1.1", 443, False),
        ],
    }
    failed = False
    for network, checks in networks.items():
        result = subprocess.run([
            "docker", "run", "--rm", "--network", network, "--read-only",
            "--cap-drop", "ALL", "--security-opt", "no-new-privileges:true",
            "-e", "CHECKS=" + json.dumps(checks), image, "python", "-c", PROBE,
        ], capture_output=True, text=True)
        print(json.dumps({"network": network, "returncode": result.returncode,
                          "checks": json.loads(result.stdout or "[]")}))
        failed |= result.returncode != 0
    internet = subprocess.run([
        "docker", "run", "--rm", "--network", "sc35-first-researchers", "--read-only",
        "--cap-drop", "ALL", "--security-opt", "no-new-privileges:true", image,
        "python", "-c",
        "import urllib.request; "
        "p=urllib.request.ProxyHandler({'https':'http://internet-proxy:3128'}); "
        "r=urllib.request.build_opener(p).open('https://github.com',timeout=30); "
        "print(r.status)",
    ], capture_output=True, text=True)
    print(json.dumps({"internet_via_proxy": internet.returncode == 0,
                      "status": internet.stdout.strip()}))
    return int(failed or internet.returncode != 0)


if __name__ == "__main__":
    sys.exit(main())
