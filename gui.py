"""Streamlit GUI for Loom - tree-based text exploration."""

import streamlit as st
from pathlib import Path
from api import Loom, TREES_DIR

st.set_page_config(page_title="Loom", page_icon="🧵", layout="wide")

# Get query params for persistence across refresh
params = st.query_params

# Initialize session state
if "loom" not in st.session_state:
    st.session_state.loom = Loom()
if "last_branches" not in st.session_state:
    st.session_state.last_branches = []
if "message" not in st.session_state:
    st.session_state.message = ""
if "last_mtime" not in st.session_state:
    st.session_state.last_mtime = 0

loom = st.session_state.loom

# Restore from query params on refresh
if "file" in params and "node" in params:
    file_name = params["file"]
    node_id = params["node"]
    # Check if we need to load the file
    watch_path = TREES_DIR / file_name
    if watch_path.exists():
        current_mtime = watch_path.stat().st_mtime
        if current_mtime != st.session_state.last_mtime:
            loom.load(file_name)
            st.session_state.last_mtime = current_mtime
        # Restore node position
        node = loom.tree.get_node(node_id)
        if node:
            loom.current_node = node

def save_position():
    """Save current position to query params."""
    if hasattr(st.session_state, 'current_file') and st.session_state.current_file:
        st.query_params["file"] = st.session_state.current_file
        st.query_params["node"] = loom.current_node.id

def get_siblings():
    """Get previous and next siblings of current node."""
    if not loom.current_node.parent_id:
        return None, None
    parent = loom.tree.get_node(loom.current_node.parent_id)
    if not parent:
        return None, None
    siblings = parent.children
    idx = next((i for i, c in enumerate(siblings) if c.id == loom.current_node.id), -1)
    prev_sib = siblings[idx - 1] if idx > 0 else None
    next_sib = siblings[idx + 1] if idx < len(siblings) - 1 else None
    return prev_sib, next_sib

def render_tree_html(node, prefix="", is_last=True, current_id=None, params=None):
    """Render tree as HTML with clickable nodes."""
    preview = node.text[:40].replace('\n', ' ').replace('<', '&lt;').replace('>', '&gt;') if node.text else "[empty]"
    if node.text and len(node.text) > 40:
        preview += "..."
    is_current = node.id == current_id

    # Build the tree prefix
    if prefix == "":
        connector = ""
        display_prefix = ""
    else:
        connector = "└── " if is_last else "├── "
        display_prefix = prefix

    # Style for current vs other nodes
    if is_current:
        style = "color: #00CED1; font-weight: bold;"
        marker = "▶"
    else:
        style = "color: #888; cursor: pointer;"
        marker = "○"
        style += " text-decoration: underline;"

    node_html = f'<span style="{style}" data-node-id="{node.id}">{marker} {preview}</span>'
    line = f'<div style="font-family: monospace; white-space: pre; line-height: 1.4;">{display_prefix}{connector}{node_html}</div>'

    lines = [line]

    # Prepare prefix for children
    if prefix == "":
        child_prefix = ""
    else:
        child_prefix = prefix + ("    " if is_last else "│   ")

    for i, child in enumerate(node.children):
        is_last_child = (i == len(node.children) - 1)
        lines.extend(render_tree_html(child, child_prefix, is_last_child, current_id, params))

    return lines


def collect_node_ids(node):
    """Collect all node IDs in the tree."""
    ids = [node.id]
    for child in node.children:
        ids.extend(collect_node_ids(child))
    return ids


# Sidebar
with st.sidebar:
    st.title("🧵 Loom")

    # File operations
    st.subheader("Files")

    TREES_DIR.mkdir(exist_ok=True)
    saved_files = sorted(TREES_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    file_options = ["New Tree"] + [f.name for f in saved_files]

    # Default to current file if set
    default_idx = 0
    if "file" in params and params["file"] in file_options:
        default_idx = file_options.index(params["file"])

    selected_file = st.selectbox("Load tree", file_options, index=default_idx, key="load_select")

    if st.button("Load", use_container_width=True):
        if selected_file != "New Tree":
            result = loom.load(selected_file)
            st.session_state.current_file = selected_file
            st.session_state.last_mtime = (TREES_DIR / selected_file).stat().st_mtime
            st.query_params["file"] = selected_file
            st.query_params["node"] = loom.current_node.id
            st.session_state.message = result
            st.rerun()
        else:
            st.session_state.loom = Loom()
            loom = st.session_state.loom
            if "file" in st.query_params:
                del st.query_params["file"]
            if "node" in st.query_params:
                del st.query_params["node"]
            st.rerun()

    if st.button("Save", use_container_width=True):
        result = loom.save()
        if "Saved to" in result:
            saved_path = Path(result.split("Saved to ")[-1])
            st.session_state.current_file = saved_path.name
            st.session_state.last_mtime = saved_path.stat().st_mtime
            st.query_params["file"] = saved_path.name
            st.query_params["node"] = loom.current_node.id
        st.session_state.message = result

    if "file" in params:
        st.caption(f"📁 {params['file']}")

    # Navigation
    st.subheader("Navigation")

    nav_cols = st.columns(2)
    with nav_cols[0]:
        if st.button("⬆️ Up", use_container_width=True, key="nav_up"):
            loom.up()
            if "file" in params:
                st.query_params["node"] = loom.current_node.id
            st.rerun()
    with nav_cols[1]:
        if st.button("🏠 Root", use_container_width=True, key="nav_root"):
            loom.root()
            if "file" in params:
                st.query_params["node"] = loom.current_node.id
            st.rerun()

    # Sibling navigation
    prev_sib, next_sib = get_siblings()
    sib_cols = st.columns(2)
    with sib_cols[0]:
        if st.button("◀️ Prev", use_container_width=True, disabled=prev_sib is None, key="nav_prev"):
            if prev_sib:
                loom.current_node = prev_sib
                if "file" in params:
                    st.query_params["node"] = loom.current_node.id
                st.rerun()
    with sib_cols[1]:
        if st.button("Next ▶️", use_container_width=True, disabled=next_sib is None, key="nav_next"):
            if next_sib:
                loom.current_node = next_sib
                if "file" in params:
                    st.query_params["node"] = loom.current_node.id
                st.rerun()

    # Children navigation
    if loom.current_node.children:
        st.write("Children:")
        for i, child in enumerate(loom.current_node.children):
            preview = child.text[:30].replace('\n', ' ') if child.text else "[empty]"
            if len(child.text or "") > 30:
                preview += "..."
            if st.button(f"[{i+1}] {preview}", key=f"child_{i}", use_container_width=True):
                loom.child(i + 1)
                if "file" in params:
                    st.query_params["node"] = loom.current_node.id
                st.rerun()

    # Settings
    st.subheader("Settings")
    temp = st.slider("Temperature", 0.1, 2.0, loom.generator.config.temperature, 0.1)
    loom.generator.config.temperature = temp

    max_tokens = st.slider("Max tokens", 20, 300, loom.generator.config.max_tokens, 10)
    loom.generator.config.max_tokens = max_tokens

    # Tree view
    st.subheader("Tree Structure")
    tree_html = "\n".join(render_tree_html(loom.tree.root, current_id=loom.current_node.id))
    st.markdown(tree_html, unsafe_allow_html=True)

    # Navigation dropdown
    all_node_ids = collect_node_ids(loom.tree.root)
    node_options = {}
    for nid in all_node_ids:
        n = loom.tree.get_node(nid)
        if n:
            preview = n.text[:30].replace('\n', ' ') if n.text else "[empty]"
            if len(n.text or "") > 30:
                preview += "..."
            node_options[nid] = f"{nid[:8]}: {preview}"

    current_idx = all_node_ids.index(loom.current_node.id) if loom.current_node.id in all_node_ids else 0
    selected = st.selectbox(
        "Jump to node:",
        all_node_ids,
        index=current_idx,
        format_func=lambda x: node_options.get(x, x),
        key="node_select"
    )
    if selected != loom.current_node.id:
        target_node = loom.tree.get_node(selected)
        if target_node:
            loom.current_node = target_node
            if "file" in params:
                st.query_params["node"] = selected
            st.rerun()

# Main content
col_left, col_right = st.columns([2, 1])

with col_left:
    st.subheader("Current Text")

    path = loom.tree.get_path_to_node(loom.current_node.id)
    prefix_text = "".join(n.text for n in path[:-1])
    current_text = loom.current_node.text or ""

    if prefix_text:
        st.markdown(f'<div style="color: #888; white-space: pre-wrap;">{prefix_text}</div>', unsafe_allow_html=True)
    if current_text:
        st.markdown(f'<div style="color: #00CED1; font-weight: bold; white-space: pre-wrap;">{current_text}</div>', unsafe_allow_html=True)

    if not prefix_text and not current_text:
        st.info("Empty tree. Write some text to start.")

    st.caption(f"Node: {loom.current_node.id} | Depth: {loom.depth} | Children: {len(loom.current_node.children)}")

    if loom.current_node.analysis:
        with st.expander("📊 Analysis", expanded=True):
            st.markdown(loom.current_node.analysis)

with col_right:
    st.subheader("Actions")

    with st.expander("✍️ Write Text", expanded=not bool(loom.current_node.text)):
        new_text = st.text_area("Enter text:", height=150, key="write_input")
        if st.button("Add Text", use_container_width=True):
            if new_text.strip():
                loom.write(new_text)
                if "file" in params:
                    st.query_params["node"] = loom.current_node.id
                st.rerun()

    with st.expander("🌿 Generate", expanded=bool(loom.current_node.text)):
        num_branches = st.number_input("Branches", 1, 5, 3, key="num_branches")

        gcol1, gcol2 = st.columns(2)
        with gcol1:
            if st.button("Generate", use_container_width=True):
                with st.spinner("Generating..."):
                    loom.generate(n=num_branches)
                    st.session_state.last_branches = loom._last_branches.copy()
                st.rerun()
        with gcol2:
            if st.button("Continue", use_container_width=True):
                with st.spinner("Continuing..."):
                    loom.continue_branch()
                    if "file" in params:
                        st.query_params["node"] = loom.current_node.id
                st.rerun()

        if st.session_state.last_branches:
            st.write("**Generated branches:**")
            for i, text in enumerate(st.session_state.last_branches):
                preview = text[:150].replace('\n', ' ')
                if len(text) > 150:
                    preview += "..."
                st.markdown(f"**[{i+1}]** {preview}")
                if st.button(f"Select {i+1}", key=f"select_{i}", use_container_width=True):
                    loom.select(i + 1)
                    st.session_state.last_branches = []
                    if "file" in params:
                        st.query_params["node"] = loom.current_node.id
                    st.rerun()

            if st.button("Add All Branches", use_container_width=True):
                loom.select_all()
                st.session_state.last_branches = []
                st.rerun()

    with st.expander("✂️ Split/Trim"):
        if loom.current_node.text:
            st.caption(f"Current text: {len(loom.current_node.text)} chars")
            split_at = st.number_input("Split at character:", 1, len(loom.current_node.text) - 1,
                                       min(100, len(loom.current_node.text) - 1), key="split_at")
            st.caption(f"Preview: ...{loom.current_node.text[max(0, split_at-20):split_at]}|{loom.current_node.text[split_at:split_at+20]}...")

            tcol1, tcol2 = st.columns(2)
            with tcol1:
                if st.button("Trim", use_container_width=True):
                    loom.trim(split_at)
                    st.rerun()
            with tcol2:
                if st.button("Split (keep)", use_container_width=True):
                    loom.split(split_at, keep_remainder=True)
                    st.rerun()
        else:
            st.info("No text to split")

    with st.expander("🔍 Analyze"):
        if st.button("Analyze with Claude", use_container_width=True):
            with st.spinner("Analyzing..."):
                loom.analyze()
            st.rerun()

# Status message
if st.session_state.message:
    st.success(st.session_state.message)
    st.session_state.message = ""
