import streamlit as st
import pandas as pd

from typing import List
from streamlit_searchbox import st_searchbox

if "edges" not in st.session_state:
    edges = pd.DataFrame([], columns=["source", "relationship", "target"])
    st.session_state.edges = edges

st.title("Biolink integrator")


def search_biolink(value: str) -> List[str]:
    return ["a", "b", "c"]


def search_relationship(value: str) -> List[str]:
    return [el for el in ["treats", "has_effect"] if el.startswith(value)]


def add_row():
    row = pd.DataFrame(
        [
            [
                st.session_state.source,
                st.session_state.relationship,
                st.session_state.target,
            ]
        ],
        columns=["source", "relationship", "target"],
    )
    st.session_state.edges = pd.concat([st.session_state.edges, row])


source = st_searchbox(
    search_biolink,
    key="source",
    label="Source biolink node",
    placeholder="Search here...",
)

relationship = st_searchbox(
    search_relationship,
    key="relationship",
    label="Source biolink relationship",
    placeholder="Search here...",
)

target = st_searchbox(
    search_biolink,
    key="target",
    label="Target biolink node",
    placeholder="Search here...",
)

st.button("Add", type="primary", on_click=add_row())

st.table(st.session_state.edges)
