# Third-party sources and attribution

This package does not relicense upstream work. Fixed sources used by the study:

| Component | Upstream repository | Study reference revision | Distribution here |
|---|---|---|---|
| ChannelTokenFormer | https://github.com/jinkwan1115/ChannelTokenFormer | b1c100effb9c27cd25de07e91de7f9005d18d5e7 | Not bundled; fetch from upstream after reviewing terms |
| PhyGNN | https://github.com/DiYi1999/PhyGNN | 75ab808f4a12da9eab650f17bb38f91d9a571154 | Not bundled; original CC BY-NC 4.0 terms |
| MTGNN | https://github.com/nnzhan/MTGNN | f811746fa7022ebf336f9ecd2434af5f365ecbf6 | Not bundled; MIT |
| PatchTST | https://github.com/yuqinie98/PatchTST | 204c21efe0b39603ad6e2ca640ef5896646ab1a9 | Not bundled; Apache 2.0 |

The local CTF model/layer files match the listed study revision; local README
edits are not treated as a new algorithm version. An explicit root licence was
not located in that CTF snapshot or the checked upstream page. Public visibility
does not by itself grant permission to relicense or redistribute the source.

Fetching is an explicit user action; this package does not execute clone or
download on import. The setup script refuses to overwrite an existing upstream
directory. Check attribution/notice obligations for any later redistributed copy.

The PhyGNN/MTGNN/PatchTST adapters and disclosure of their task-specific changes
are in `workspace/PhysicsRelationCTF/runs/physics_ai_comparison_v1/UPSTREAM.md`.
Their original code, datasets, compiled dependencies and entire vendor trees
are not silently included in this archive.

Physics equation source citations and direct-edge limitations are retained in
the graph/feature definitions. Public LBNL and TEP data have their own access
routes and terms described in `DATA_AVAILABILITY.md`.
