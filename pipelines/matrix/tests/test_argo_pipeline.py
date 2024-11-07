# What would be an e2e test for ArgoPipeline?


# IN: Kedro pipeline, containing ArgoNodes which are supposed to be fused. Ideally something based on the tests that are already there.

# When Argo Pipeline is initialized (with the Kedro pipeline), it should automatically fuse the nodes which need fusing.

# Then, the only thing remaining is to render the Argo Template.

# OUT: Argo template, which can be rendered to an Argo workflow.
