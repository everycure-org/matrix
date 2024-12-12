import streamlit as st
import pandas as pd
from utils import (
    FlowDiagram,
    combine_moa_predictions_and_pair_info,
    list_available_pairs_df,
    get_moa_predictions_from_db,
    build_external_urls,
    display_table,
    parse_hop_parameter,
)
from config import settings, set_wide_space_default

set_wide_space_default()

with st.sidebar:
    col1, col2 = st.columns(
        [0.6, 0.4],
        gap="small",
    )
    st.title("Available pairs")
    st.write(
        "Select the pair you would like to view from the list of", " available pairs, or search using the search bar"
    )

    # Search bar
    with col1:
        search_input = st.text_input("Search for by entity ID", key="search_input", value="")
    # Selector number of hops
    with col2:
        hop_selector = st.toggle("2/3 hops", value=False, key="hop_selector")
        if not hop_selector:
            st.write("Viewing 2-hop paths")
        else:
            st.write("Viewing 3-hop paths")
        hop_selector_parsed = parse_hop_parameter(hop_selector)

    available_pairs_df = list_available_pairs_df(path_number=hop_selector_parsed)

    if search_input != "":
        if "|" in search_input:
            available_pairs_df = available_pairs_df[
                available_pairs_df["pair_id"].str.contains(search_input, regex=False)
            ]
        else:
            available_pairs_df = available_pairs_df[
                available_pairs_df["pair_id"].str.contains(f"({search_input}\||{search_input}$)", regex=True)
            ]

    all_pairs = st.checkbox("Show all pairs", value=False)
    if all_pairs or search_input != "":
        selected_pair_index_df = st.dataframe(
            display_table(available_pairs_df), selection_mode="single-row", on_select="rerun", hide_index=True
        )

with st.expander("What is the MoA prediction pipeline?"):
    st.markdown(settings.moa_info_text)
    st.image(settings.moa_info_img)

if "selected_pair_index_df" in locals():
    if len(selected_pair_index_df.selection.rows) > 0:
        # Show user which pair they are looking at
        index = selected_pair_index_df.selection.rows[0]
        selected_pair_id = available_pairs_df.iloc[index]["pair_id"]
        selected_disease_name = available_pairs_df.iloc[index]["disease_name"]
        selected_drug_name = available_pairs_df.iloc[index]["drug_name"]
        st.write(f"You are viewing pair: {selected_drug_name}---{selected_disease_name}")

        # Get MoA predictions
        moa_prediction_df = get_moa_predictions_from_db(
            input_path=settings.moa_db_path, pair_id=selected_pair_id, path_number=hop_selector_parsed
        )

        moa_prediction_df["protein_url"] = moa_prediction_df["intermediate_id_1"].apply(build_external_urls)

        moa_prediction_df = combine_moa_predictions_and_pair_info(
            moa_predictions=moa_prediction_df, pair_info=available_pairs_df.iloc[[index]]
        )
        moa_prediction_df["Feedback"] = ""

        # Display table representation of paths
        st.data_editor(
            display_table(moa_prediction_df),
            column_config={
                # 'protein_url': st.column_config.LinkColumn(
                #     display_text=r"https://www.uniprot.org/uniprotkb/(.*?)/entry"
                # ),
                # "Feedback": st.column_config.SelectboxColumn(help="Does this MoA make sense?", options=["Yes", "No"])
            },
        )
        if "curr_state" not in st.session_state:
            # Set dummy current state
            st.session_state.curr_state = ""

        # Display graphical representation of paths
        flow_diagram = FlowDiagram()
        graph_states = []
        for index, row in moa_prediction_df.iterrows():
            single_row_df = pd.DataFrame([row])
            graph_states.append(flow_diagram.create_graph_state(single_row_df))

        states = []
        interactivity = False
        for i, graph_state in enumerate(graph_states):
            states.append(
                flow_diagram.render_graph(
                    graph_state=graph_states[i], graph_label=f"{i}_graph", interactivity=interactivity
                )
            )
