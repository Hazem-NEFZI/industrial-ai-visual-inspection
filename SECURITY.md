# Input and artifact handling

Validate image sizes, schemas, and numerical finiteness before inference. Keep uploaded content local unless an explicit deployment design requires otherwise. Do not load untrusted pickle/joblib files. PyTorch checkpoints use weights-only loading where supported. This project is not a security-hardened multi-tenant service.

Report problems privately to the repository maintainer using the contact method they publish; no security contact is invented in this template.
