"""Pull a Docker Hub image without docker and unpack it into a chroot-able rootfs.
  python3 atom_image_fetch.py [--repo rocm/atom-dev] [--tag nightly_202609250902] [--out /shared_nfs/kk/atom_image]
Out: <out>/blobs/<sha>, <out>/config.json (image ENV/Cmd), <out>/rootfs/ ; progress lines on stdout."""
import argparse, concurrent.futures as cf, hashlib, json, os, shutil, subprocess, sys, time, urllib.request

ap = argparse.ArgumentParser()
ap.add_argument("--repo", default="rocm/atom-dev")
ap.add_argument("--tag", default="nightly_202609250902")
ap.add_argument("--out", default="/shared_nfs/kk/atom_image")
ap.add_argument("--jobs", type=int, default=8)
a = ap.parse_args()
REG = "https://registry-1.docker.io/v2/" + a.repo
blobs = os.path.join(a.out, "blobs"); root = os.path.join(a.out, "rootfs")
os.makedirs(blobs, exist_ok=True)


def token():
    u = f"https://auth.docker.io/token?service=registry.docker.io&scope=repository:{a.repo}:pull"
    return json.load(urllib.request.urlopen(u, timeout=30))["token"]


def get(path, accept=None, stream_to=None):
    req = urllib.request.Request(REG + path, headers={"Authorization": "Bearer " + token(),
                                                      **({"Accept": accept} if accept else {})})
    r = urllib.request.urlopen(req, timeout=120)
    if stream_to is None:
        return r.read()
    with open(stream_to, "wb") as f:
        shutil.copyfileobj(r, f, 1 << 22)


man = json.loads(get("/manifests/" + a.tag, "application/vnd.docker.distribution.manifest.v2+json,"
                                             "application/vnd.oci.image.manifest.v1+json"))
open(os.path.join(a.out, "manifest.json"), "w").write(json.dumps(man, indent=1))
cfg = json.loads(get("/blobs/" + man["config"]["digest"]))
open(os.path.join(a.out, "config.json"), "w").write(json.dumps(cfg, indent=1))
layers = man["layers"]
print(f"layers={len(layers)} GB={sum(l['size'] for l in layers)/1e9:.1f} types={sorted({l['mediaType'].split('.')[-1] for l in layers})}", flush=True)


def fetch(l):
    dg = l["digest"]; p = os.path.join(blobs, dg.split(":")[1])
    for attempt in range(5):
        if os.path.exists(p) and os.path.getsize(p) == l["size"]:
            h = hashlib.sha256()
            with open(p, "rb") as f:
                for b in iter(lambda: f.read(1 << 24), b""):
                    h.update(b)
            if "sha256:" + h.hexdigest() == dg:
                return p
            os.remove(p)
        try:
            get("/blobs/" + dg, stream_to=p + ".part"); os.replace(p + ".part", p)
        except Exception as e:
            print(f"retry {dg[:19]} {attempt}: {e}", flush=True); time.sleep(5)
    raise RuntimeError("failed " + dg)


t0 = time.time(); done = 0
with cf.ThreadPoolExecutor(a.jobs) as ex:
    for _ in ex.map(fetch, layers):
        done += 1
        if done % 5 == 0 or done == len(layers):
            print(f"downloaded {done}/{len(layers)} in {time.time()-t0:.0f}s", flush=True)

os.makedirs(root, exist_ok=True)
for i, l in enumerate(layers):
    p = os.path.join(blobs, l["digest"].split(":")[1])
    z = "--zstd" if "zstd" in l["mediaType"] else "-z" if "gzip" in l["mediaType"] else ""
    names = subprocess.run(f"tar {z} -tf {p}", shell=True, capture_output=True, text=True, check=True).stdout.split("\n")
    for n in names:
        base = os.path.basename(n)
        if not base.startswith(".wh."):
            continue
        d = os.path.join(root, os.path.dirname(n))
        if base == ".wh..wh..opq":
            if os.path.isdir(d):
                for e in os.listdir(d):
                    q = os.path.join(d, e)
                    shutil.rmtree(q) if os.path.isdir(q) and not os.path.islink(q) else os.remove(q)
        else:
            q = os.path.join(d, base[4:])
            if os.path.lexists(q):
                shutil.rmtree(q) if os.path.isdir(q) and not os.path.islink(q) else os.remove(q)
    subprocess.run(f"tar {z} -xf {p} -C {root} --exclude='.wh.*' --numeric-owner",
                   shell=True, check=True)
    print(f"extracted {i+1}/{len(layers)}", flush=True)
print("ROOTFS_READY", root, f"{time.time()-t0:.0f}s", flush=True)
