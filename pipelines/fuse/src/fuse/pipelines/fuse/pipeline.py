"""
This is a boilerplate pipeline 'fuse'
generated using Kedro 0.19.7
"""

from typing import Optional

from kedro.pipeline import Pipeline, node, pipeline

from fuse.runners.fuse_runner import FusedPipeline


def dummy(arg: Optional[str] = None):
    return {"foo": arg}


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=dummy,
                inputs=[],
                outputs="fuse.foo",
                name="foo",
            ),
            FusedPipeline(
                [
                    node(
                        func=dummy,
                        inputs=["fuse.foo"],
                        outputs="fuse.first",
                        name="first",
                    ),
                    node(
                        func=dummy,
                        inputs=["fuse.first"],
                        outputs="fuse.second",
                        name="second",
                    ),
                ],
                name="example",
            ),
            node(
                func=dummy,
                inputs=["fuse.second"],
                outputs="fuse.bar",
                name="bar",
            ),
        ]
    )
