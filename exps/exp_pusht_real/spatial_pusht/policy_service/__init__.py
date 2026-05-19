"""HTTP service that exposes a trained diffusion-policy checkpoint.

Run independently of the viser inference coordinator so the GPU-heavy
model lives in its own process (and its own conda env). The coordinator
talks to it over JSON HTTP, mirroring how `pusht_service` exposes the
real arm.
"""
