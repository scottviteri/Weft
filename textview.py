"""Build the interactive Current Text view for the GUI.

Kept separate from gui.py (which can only run inside Streamlit) so the
HTML-assembly logic is importable and unit-testable on its own. The returned
HTML is dropped into a Streamlit `components.html` iframe; words carry these
interactions, wired by the embedded JS:

  - hover           -> highlight the matching plot point + show next-token candidates
  - click / shift   -> cycle to next / previous sibling at a branch point
  - alt-click       -> split here and open a new sibling branch

Sibling/split actions reach the app by setting a `goto`/`splitat` query param
on the parent window (the same channel gui.py uses to persist node position).
"""

import html
import json

from coloring import (
    token_segments, hex_for_logprob, token_title, alt_bar_items, logprob_plot_svg,
)

_TEMPLATE = """<!doctype html><html><head><meta charset="utf-8"><style>
 body{margin:0;font-family:'Source Sans Pro',system-ui,sans-serif;background:transparent;color:#9aa0a6;}
 .text{white-space:pre-wrap;line-height:1.75;font-size:1.02rem;}
 .text span{border-radius:2px;}
 .cur{font-weight:bold;}
 /* Visible 8px pill, but content-box padding makes the click target ~18px. */
 .fork{display:inline-block;box-sizing:content-box;width:8px;height:1em;
   padding:2px 5px;margin:0 1px;vertical-align:-0.2em;cursor:pointer;
   background:#6aa3ff;background-clip:content-box;border-radius:3px;
   box-shadow:0 0 4px rgba(106,163,255,.7);transition:background .12s,box-shadow .12s;}
 .fork:hover{background:#ffd23c;background-clip:content-box;box-shadow:0 0 7px rgba(255,210,60,.95);}
 .fork:active{background:#ffb300;background-clip:content-box;}
 .hl{background:rgba(255,221,0,.45);}
 .altbar{min-height:104px;margin-top:10px;padding-top:8px;border-top:1px solid rgba(136,136,136,.25);}
 .alttitle{font-size:.7rem;color:#888;margin-bottom:5px;text-transform:uppercase;letter-spacing:.05em;}
 .altrow{display:flex;align-items:center;gap:7px;margin:1px 0;font-size:.8rem;}
 .altlabel{width:84px;flex:0 0 84px;font-family:ui-monospace,monospace;white-space:pre;overflow:hidden;text-overflow:ellipsis;text-align:right;color:#bbb;}
 .altrow.chosen .altlabel{color:#fff;font-weight:bold;}
 .altfill{height:10px;background:#6aa3ff;border-radius:2px;}
 .altrow.chosen .altfill{background:#ffd23c;}
 .altpct{color:#888;font-variant-numeric:tabular-nums;}
 .pt{cursor:pointer;transition:r .1s;}
 .plot{margin-top:10px;border-top:1px solid rgba(136,136,136,.25);padding-top:8px;}
 .hint{font-size:.7rem;color:#777;margin-top:8px;}
</style></head><body>
 <div class="text">__TEXT__</div>
 <div class="altbar" id="altbar"><div class="alttitle">hover a word for its next-token candidates</div></div>
 <div class="plot">__SVG__</div>
 <div class="hint">blue bar = branch point (between the shared and diverging token) · click cycles siblings, shift-click reverses · alt-click a word to split a new branch there</div>
 <script>
 var ALTS=__ALTS__;
 var APP=__APP__;
 var CNT=0;
 function esc(s){var d=document.createElement('div');d.textContent=s;return d.innerHTML;}
 function disp(s){return s.replace(/\\n/g,'\\u23ce').replace(/ /g,'\\u00b7')||'\\u2205';}
 function setPlot(i,on){var p=document.getElementById('p'+i);if(p){p.setAttribute('r',on?6:3);p.style.strokeWidth=on?2:0;}}
 function spanFor(i){return document.querySelector('.text span[data-i="'+i+'"]');}
 function showAlts(i){
   var a=ALTS[i],bar=document.getElementById('altbar');
   if(!a){return;}
   var max=a[0][1]||1;
   bar.innerHTML='<div class="alttitle">next-token candidates</div>'+a.map(function(r){
     var w=Math.max(2,Math.round(r[1]/max*150));
     return '<div class="altrow'+(r[2]?' chosen':'')+'"><span class="altlabel">'+esc(disp(r[0]))+'</span>'
       +'<span class="altfill" style="width:'+w+'px"></span>'
       +'<span class="altpct">'+(r[1]*100).toFixed(1)+'%</span></div>';
   }).join('');
 }
 function enter(i){var s=spanFor(i);if(s){s.classList.add('hl');}setPlot(i,true);showAlts(i);}
 function leave(i){var s=spanFor(i);if(s){s.classList.remove('hl');}setPlot(i,false);}
 // Preferred channel: write the action into a hidden Streamlit text_input in
 // the parent (allow-same-origin lets us reach it) and commit it, which reruns
 // the app over the existing WebSocket — no full page reload. A counter keeps
 // each command unique so repeats still register.
 function sendCmd(kind,value){
   try{
     var pdoc=window.parent.document;
     var input=pdoc.querySelector('input[aria-label="weft_cmd"]');
     if(input){
       CNT++;
       var setter=Object.getOwnPropertyDescriptor(window.parent.HTMLInputElement.prototype,'value').set;
       setter.call(input,kind+':'+value+':'+CNT);
       input.dispatchEvent(new window.parent.Event('input',{bubbles:true}));
       input.dispatchEvent(new window.parent.KeyboardEvent('keydown',{key:'Enter',keyCode:13,which:13,bubbles:true}));
       // Self-heal: a successful commit reruns the app and reloads this iframe,
       // destroying the timer. If nothing reran (commit didn't take), fall back
       // to the full-reload navigation after a short grace period.
       setTimeout(function(){navFallback(kind==='split'?'splitat':'goto', value);}, 700);
       return;
     }
   }catch(e){}
   navFallback(kind==='split'?'splitat':'goto', value);  // full-reload fallback
 }
 // Fallback: the iframe sandbox lacks allow-top-navigation, so we navigate by
 // injecting a <script> into the (un-sandboxed) parent that sets its location.
 function navFallback(param,value){
   var href;
   try{href=window.parent.location.href;}catch(e){href=document.referrer;}
   var u;
   try{u=new URL(href);}catch(e){return;}
   Object.keys(APP).forEach(function(k){u.searchParams.set(k,APP[k]);});
   u.searchParams.delete('goto');u.searchParams.delete('splitat');
   u.searchParams.set(param,value);
   var url=u.toString();
   try{
     var pdoc=window.parent.document;
     var s=pdoc.createElement('script');
     s.textContent='location.href='+JSON.stringify(url);
     pdoc.head.appendChild(s);pdoc.head.removeChild(s);return;
   }catch(e){}
   try{
     var pd=window.parent.document;
     var a=pd.createElement('a');a.href=url;a.target='_self';a.style.display='none';
     pd.body.appendChild(a);a.click();pd.body.removeChild(a);return;
   }catch(e){}
   try{window.top.location.href=url;}catch(e){}
 }
 // Tokens: hover to highlight + show candidates; alt-click to split here.
 document.querySelectorAll('.text span[data-i]').forEach(function(s){
   var i=s.dataset.i;
   s.addEventListener('mouseenter',function(){enter(i);});
   s.addEventListener('mouseleave',function(){leave(i);});
   s.addEventListener('click',function(e){
     if(e.altKey && s.dataset.split!==undefined){
       e.preventDefault();
       var off=parseInt(s.dataset.split,10);
       if(off>0){sendCmd('split',off);}
     }
   });
 });
 // Fork markers (between the shared and diverging token): click to cycle siblings.
 document.querySelectorAll('.fork').forEach(function(f){
   f.addEventListener('click',function(e){
     e.preventDefault();
     var sibs=(f.dataset.sibs||'').split(',').filter(Boolean);
     if(sibs.length<2){return;}
     var pos=parseInt(f.dataset.pos||'0',10);
     var next=(pos+(e.shiftKey?-1:1)+sibs.length)%sibs.length;
     sendCmd('goto',sibs[next]);
   });
 });
 document.querySelectorAll('.pt').forEach(function(p){
   var i=p.dataset.idx;
   p.addEventListener('mouseenter',function(){enter(i);});
   p.addEventListener('mouseleave',function(){leave(i);});
 });
 </script></body></html>"""


def build_text_component(loom, app_params=None):
    """Return (html, has_logprobs) for the path root..current_node.

    `app_params` is the current Streamlit query-param dict (file/node); it is
    embedded so the in-iframe JS can rebuild the parent URL when a word click
    navigates (the sandbox blocks reading the parent location directly).

    Colors every token by surprisal, inserts a clickable fork marker between the
    shared token and the diverging token at each branch point, attaches per-token
    next-token candidates for the hover bar, and links current-node tokens to the
    plot via a single global token index.
    """
    path = loom.tree.get_path_to_node(loom.current_node.id)
    spans = []
    alts = {}            # global token index -> [[token, prob, is_chosen], ...]
    plot_points = []     # (global index, logprob) for the plot window
    gidx = 0
    any_lp = False

    # The plot covers from just before the most recent branch point (the deepest
    # fork on the path) through the current node, so you can see the logprob
    # trajectory leading into and out of the fork rather than only this node.
    branch_js = [j for j, n in enumerate(path) if len(loom.siblings(n.id)) > 1]
    if branch_js:
        plot_start_j = max(0, max(branch_js) - 1)
        branch_node = path[max(branch_js)]
    else:
        plot_start_j = len(path) - 1
        branch_node = None
    mark_idx = None

    for j, node in enumerate(path):
        is_current = node is loom.current_node
        in_plot = j >= plot_start_j
        if branch_node is not None and node is branch_node:
            mark_idx = gidx  # global index of the fork's first token
        segs = token_segments(node.text, node.logprobs)
        top = (node.logprobs or {}).get("top_logprobs")
        aligned_top = top if (top and len(top) == len(segs)) else None

        sibs = None
        pos = 0
        sib_nodes = loom.siblings(node.id)
        if len(sib_nodes) > 1:
            sibs = [c.id for c in sib_nodes]
            pos = next((j for j, c in enumerate(sib_nodes) if c.id == node.id), 0)

        char_off = 0
        for k, (tok, lp) in enumerate(segs):
            classes = []
            styles = ""
            title = ""
            attrs = f'data-i="{gidx}"'

            if lp is not None:
                any_lp = True
                styles = f"color:{hex_for_logprob(lp)}"
                title = token_title(lp)
                if in_plot:
                    plot_points.append((gidx, lp))

            if is_current:
                classes.append("cur")
                attrs += f' data-split="{char_off}"'

            if aligned_top and aligned_top[k]:
                items = alt_bar_items(aligned_top[k])
                if items:
                    alts[gidx] = [[t, p, (t == tok)] for t, p in items]

            # A fork marker sits BETWEEN the shared (parent) token and this
            # node's first (diverging) token, rather than recoloring the token.
            if sibs is not None and k == 0:
                ftitle = (f"branch point · {len(sibs)} options · "
                          f"click cycles siblings, shift-click reverses")
                spans.append(
                    f'<span class="fork" data-sibs="{",".join(sibs)}" '
                    f'data-pos="{pos}" title="{html.escape(ftitle)}"></span>'
                )

            cls = f' class="{" ".join(classes)}"' if classes else ""
            sty = f' style="{styles}"' if styles else ""
            ttl = f' title="{html.escape(title)}"' if title else ""
            spans.append(f"<span{cls}{sty}{ttl} {attrs}>{html.escape(tok)}</span>")
            char_off += len(tok)
            gidx += 1

    component = (_TEMPLATE
                 .replace("__TEXT__", "".join(spans))
                 .replace("__SVG__", logprob_plot_svg(plot_points, mark_idx=mark_idx))
                 .replace("__ALTS__", json.dumps(alts).replace("</", "<\\/"))
                 .replace("__APP__", json.dumps(app_params or {})))
    return component, any_lp
