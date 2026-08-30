# Maintainer release checklist

Complete these items before making the repository public:

- [x] Set the repository URL to `KaiKai0714/layout-abi-lab`.
- [x] Set the current sole author to `Sheng-Kai Ku`.
- [x] Add the public contact email `ethankai0714@gmail.com`.
- [ ] Add an ORCID, if desired.
- [ ] Enable GitHub private vulnerability reporting after repository creation.
- [x] Add the current project copyright notice.
- [x] Re-run and strictly validate the reference protocol through the public v0.1 schema.
- [x] Preserve both the legacy E170 reference and the public v0.1 L40S bundle.
- [x] Pin a minimal set of driver-compatible images in `containers/matrix.json`.
- [x] Run and strictly validate three software stacks on L40S.
- [x] Scan release files for credentials, hostnames, and private absolute paths. The
      preserved legacy JSON contains only the synthetic `e170_benchmark` container user
      and its `/tmp` cache path.
- [x] Preserve the upstream MIT notice and document that the cited CMB repository has
      no visible license; no CMB code or data is redistributed.
- [ ] Create a signed `v0.1.0` tag and archive its generated checksums.
