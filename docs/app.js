(function () {
  "use strict";

  var TITLES = {
    A1:"Load, tokenize, and count the corpus", A2:"Normalize words for comparison",
    A3:"Build and tune TF–IDF", A4:"Build a reproducible three-word AND query",
    A5:"Score with Boolean retrieval", A6:"Score with TF–IDF cosine similarity",
    A7:"Compare a count-vector representation", A8:"Test ordered bigram evidence",
    A9:"Expand each query concept with WordNet", A10:"Consolidate the retrieval comparisons",
    A11:"Represent grammar with POS vectors", A12:"Compare documents by grammatical shape",
    B1:"Navigate the WordNet hierarchy", B2:"Inspect synonyms, antonyms, and frequency",
    B3:"Measure potato ↔ tiger in WordNet", B4:"Compare lexical-relation groups",
    B5:"Compare potato ↔ tiger with Word2Vec", B6:"Contrast FastText, GloVe, and BERT",
    B7:"Compare a dwelling family in WordNet", B8:"Compare the same family with embeddings",
    B9:"Ask whether the systems agree"
  };
  var ANSWERS = {
    A1:"BeautifulSoup extracts five article bodies; NLTK then tokenizes each body.",
    A2:"The tokens are lowercased, filtered, POS-tagged, and lemmatized after stopword removal.",
    A3:"The chosen TF–IDF model uses min_df=2 and max_df=0.8 after four threshold combinations are compared.",
    A4:"A fixed random seed selects three distinct raw words from D1.",
    A5:"A document scores 1 only when it contains all three Boolean AND query terms.",
    A6:"The fitted TF–IDF model transforms the query before cosine similarity is calculated.",
    A7:"Raw unigram counts replace TF–IDF weights before cosine similarity is recalculated.",
    A8:"Word bigrams represent ordered pairs of adjacent tokens.",
    A9:"Up to three same-POS WordNet synonyms are added to each query term.",
    A10:"The expanded query uses OR within each synonym group and AND between the three concepts.",
    A11:"Each document is represented by counts on 45 Penn Treebank POS-tag axes.",
    A12:"Cosine similarity is calculated between every pair of POS-count vectors.",
    B1:"The chosen noun sense, its first hypernym, and all descendant hyponyms are retrieved.",
    B2:"Synonyms and direct antonyms are ranked by WordNet lemma frequency.",
    B3:"All same-POS sense pairs are tested; the highest path and Wu–Palmer scores are reported.",
    B4:"The synonym and antonym groups are compared with both WordNet measures.",
    B5:"A 100-dimensional skip-gram Word2Vec model is trained on the Brown corpus.",
    B6:"FastText, pretrained GloVe, and mean-pooled BERT embeddings are compared with cosine similarity.",
    B7:"Explicit dwelling noun senses are compared with path and Wu–Palmer similarity.",
    B8:"Each embedding model represents the dwelling words and compares their vectors.",
    B9:"Spearman correlation compares the WordNet and embedding rankings."
  };
  var SECTION_A_RESULTS = {A1:true,A2:true,A3:true,A4:true,A10:true,A12:true};
  var state = {data:null, articles:[], activeArticle:-1};

  function esc(value) {
    return String(value == null ? "" : value).replace(/&/g,"&amp;").replace(/</g,"&lt;")
      .replace(/>/g,"&gt;").replace(/"/g,"&quot;").replace(/'/g,"&#039;");
  }
  function fmt(value,digits) {
    if(value==null) return "—";
    return typeof value==="number" ? value.toLocaleString(undefined,{maximumFractionDigits:digits==null?3:digits}) : esc(value);
  }
  function metric(value,label) {
    return '<div class="metric"><strong>'+fmt(value)+'</strong><span>'+esc(label)+'</span></div>';
  }
  function chips(items,muted) {
    return '<div class="chips">'+(items||[]).map(function(x){
      return '<span class="chip'+(muted?' muted':'')+'">'+esc(x)+'</span>';
    }).join("")+'</div>';
  }
  function table(headers,rows) {
    return '<div class="table-wrap"><table><thead><tr>'+headers.map(function(x){return '<th>'+esc(x)+'</th>';}).join("")+
      '</tr></thead><tbody>'+rows.map(function(row){return '<tr>'+row.map(function(x){return '<td>'+fmt(x)+'</td>';}).join("")+'</tr>';}).join("")+
      '</tbody></table></div>';
  }
  function heatmap(labels,matrix) {
    var cells=['<div class="heat-label"></div>'];
    labels.forEach(function(x){cells.push('<div class="heat-label">'+esc(x)+'</div>');});
    matrix.forEach(function(row,i){
      cells.push('<div class="heat-label">'+esc(labels[i])+'</div>');
      row.forEach(function(raw){
        var value=Number(raw||0),light=96-Math.max(0,Math.min(1,value))*48;
        cells.push('<div class="heat-cell" title="'+fmt(value,6)+'" style="background:hsl(156 54% '+light+
          '%);color:'+(light<66?'#fff':'#12211c')+'">'+fmt(value,2)+'</div>');
      });
    });
    return '<div class="heatmap" style="grid-template-columns:repeat('+(labels.length+1)+',minmax(50px,1fr))">'+cells.join("")+'</div>';
  }
  function code(taskId) {
    var s=state.data.code_snippets[taskId];
    return '<section class="code-panel"><div class="code-head"><h4>Implementation · '+esc(s.function)+
      '</h4><a href="'+esc(s.source_url)+'">lab2.py lines '+s.line_start+'–'+s.line_end+
      ' ↗</a></div><pre><code>'+esc(s.code)+'</code></pre></section>';
  }

  function evidenceA(id,t) {
    if(id==="A1") return '<p class="result-intro">Every raw token is listed alphabetically, including repeated occurrences.</p>'+t.map(function(a){
      return '<details class="vocabulary-list"><summary><strong>'+a.id+'</strong><span>'+fmt(a.tokens.length)+
        ' sorted tokens · vocabulary '+fmt(a.vocabulary_size)+'</span></summary>'+chips(a.tokens,true)+'</details>';
    }).join("");
    if(id==="A2") return '<p class="result-intro">These are the normalized vocabularies used by the retrieval models.</p>'+t.map(function(d){
      return '<details class="vocabulary-list"><summary><strong>'+d.id+'</strong><span>'+fmt(d.vocabulary_size)+
        ' vocabulary items · '+fmt(d.token_count)+' tokens</span></summary>'+chips(d.vocabulary,true)+'</details>';
    }).join("");
    if(id==="A3") {
      var rows=t.vectors.map(function(vector,index){
        var weighted=vector.map(function(value,i){return {term:t.features[i],value:value};})
          .filter(function(item){return item.value>0;}).sort(function(a,b){return b.value-a.value;}).slice(0,10);
        return ["D"+(index+1),weighted.map(function(item){return item.term+" "+fmt(item.value,3);}).join(" · ")];
      });
      return '<div class="metric-grid">'+metric(t.features.length,"TF–IDF dimensions")+
        metric(2,"minimum document frequency")+metric(.8,"maximum document proportion")+'</div>'+table(["Document","Highest-weight coordinates"],rows);
    }
    if(id==="A4") return '<div class="query-explainer"><span>Query Q</span><strong>'+t.original.map(esc).join(' <em>AND</em> ')+
      '</strong><p>The model uses the normalized forms shown below.</p></div>'+chips(t.normalized,true)+
      '<div class="metric-grid">'+metric(t.seed,"random seed")+metric(t.operator,"logical operator")+'</div>';
    if(id==="A10") {
      var a=state.data.section_a,original={boolean:a.A5,tfidf:a.A6,count:a.A7,bigram:a.A8},rows=[];
      [["Q · original",original],["Q1 · expanded",t]].forEach(function(group){
        Object.keys(group[1]).forEach(function(model){
          var scores=group[1][model],best=Math.max.apply(null,scores);
          rows.push([group[0],model.toUpperCase()].concat(scores).concat(["D"+(scores.indexOf(best)+1)]));
        });
      });
      return '<p class="result-intro">The final table collects the A5–A10 retrieval results.</p>'+table(["Query","Model","D1","D2","D3","D4","D5","Highest"],rows);
    }
    if(id==="A12") return heatmap(state.data.section_a.A1.map(function(a){return a.id;}),t);
    return "";
  }

  function evidenceB(id,t) {
    var words=state.data.section_b.pairs.selected;
    if(id==="B1") return '<p class="pair-label">Selected pair · '+esc(words.join(" ↔ "))+'</p><div class="metric-grid">'+words.map(function(word){
      var item=t.selected[word];
      return '<div class="metric"><strong>'+esc(word)+': '+esc(item.synset)+'</strong><span>Hypernym: '+esc(item.hypernym)+
        ' · '+item.hyponyms.length+' descendant hyponyms</span></div>';
    }).join("")+'</div><p class="note">The required house/train traversal is also present in the generated results.</p>';
    if(id==="B2") return words.map(function(word){
      var item=t.selected[word];
      return '<h4>'+word+' · ranked lemma counts</h4>'+table(["Lemma","WordNet frequency"],item.synonyms.slice(0,8).map(function(x){return [x.word,x.frequency];}))+
        '<p class="note">Direct antonyms: '+(item.antonyms.length?item.antonyms.map(function(x){return esc(x.word);}).join(", "):"none")+'.</p>';
    }).join("");
    if(id==="B3") return '<div class="metric-grid">'+metric(t.selected.maximum.path.score,"maximum path")+
      metric(t.selected.maximum.wup.score,"maximum Wu–Palmer")+
      metric(t.selected.selected.path.score,"potato.n.01 ↔ tiger.n.02 path")+
      metric(t.selected.selected.wup.score,"potato.n.01 ↔ tiger.n.02 Wu–Palmer")+'</div><div class="wordnet-tree">'+
      esc(t.selected.maximum.path.first_hierarchy.join(" → "))+'<br>'+esc(t.selected.maximum.path.second_hierarchy.join(" → "))+
      '<br>LCS: '+esc(t.selected.maximum.path.least_common_subsumers.join(", "))+'</div>';
    if(id==="B4") {
      var groups={};
      t.selected.forEach(function(row){
        if(!groups[row.group]) groups[row.group]={name:row.group,count:0,path:0,wup:0};
        var group=groups[row.group]; group.count+=1; group.path+=row.path||0; group.wup+=row.wup||0;
      });
      return table(["Relation group","Pairs","Mean path","Mean WUP"],Object.keys(groups).map(function(name){
        var group=groups[name]; return [name,group.count,group.path/group.count,group.wup/group.count];
      }))+'<p class="note">Missing antonym groups remain empty; zero rows are not invented.</p>';
    }
    if(id==="B5") return '<div class="metric-grid">'+metric(t.selected.vectors.potato.length,"dimensions")+
      metric(t.selected.similarity[0][1],"potato ↔ tiger cosine")+'</div><p><strong>First eight potato coordinates</strong></p>'+chips(t.selected.vectors.potato.slice(0,8).map(function(x){return fmt(x,4);}),true)+
      '<p class="note">Word2Vec is trained on the NLTK Brown corpus with a fixed seed.</p>';
    if(id==="B6") return '<div class="metric-grid">'+Object.keys(t.selected).map(function(name){
      var model=t.selected[name]; return metric(model.similarity[0][1],name+' cosine · '+model.vectors.potato.length+'d');
    }).join("")+'</div>';
    if(id==="B7") return '<div class="select-row"><label for="wordnet-metric">Metric</label><select id="wordnet-metric"><option value="wup">Wu–Palmer</option><option value="path">Path</option></select></div><div id="wordnet-map">'+heatmap(t.labels,t.wup)+'</div>';
    if(id==="B8") return '<div class="select-row"><label for="embedding-model">Model</label><select id="embedding-model">'+Object.keys(t).map(function(name){return '<option value="'+name+'">'+name+'</option>';}).join("")+'</select></div><div id="embedding-map">'+heatmap(state.data.section_b.B7.labels,t.word2vec.similarity)+'</div>';
    if(id==="B9") return table(["WordNet","Embedding","Spearman ρ"],t.correlations.map(function(row){return [row.wordnet,row.embedding,row.spearman_rho];}))+
      '<p class="note">'+esc(t.conclusion)+'</p>';
    return "";
  }

  function renderTasks() {
    [["A","section_a"],["B","section_b"]].forEach(function(pair){
      var letter=pair[0],source=state.data[pair[1]],target=document.querySelector("#section-"+letter.toLowerCase()+" .task-list");
      Object.keys(source).filter(function(key){return new RegExp("^"+letter+"\\d+$").test(key);}).forEach(function(id){
        var details=document.createElement("details"),show=letter==="B"||SECTION_A_RESULTS[id];
        details.className="task"; details.id="task-"+id.toLowerCase();
        details.innerHTML='<summary><span class="task-id">'+id+'</span><h3>'+esc(TITLES[id])+'</h3></summary><div class="task-body"><p class="answer">'+
          esc(ANSWERS[id])+'</p>'+(show?'<section class="evidence"><h4>Result and interpretation</h4>'+(letter==="A"?evidenceA(id,source[id]):evidenceB(id,source[id]))+'</section>':'')+code(id)+'</div>';
        target.appendChild(details);
      });
    });
  }

  function toggleArticle(index) {
    state.activeArticle=state.activeArticle===index?-1:index;
    document.querySelectorAll(".article-card-shell").forEach(function(shell,i){
      var flipped=i===state.activeArticle,button=shell.querySelector(".article-card"),link=shell.querySelector(".article-source-link");
      shell.classList.toggle("flipped",flipped); button.setAttribute("aria-pressed",flipped?"true":"false");
      button.setAttribute("aria-label",(flipped?"Show article image for ":"Show corpus statistics for ")+state.articles[i].title);
      link.setAttribute("aria-hidden",flipped?"false":"true"); link.tabIndex=flipped?0:-1;
    });
  }
  function renderArticles() {
    var rail=document.getElementById("article-rail"),normalized={};
    state.data.section_a.A2.forEach(function(item){normalized[item.id]=item;});
    state.articles.forEach(function(article,index){
      var item=normalized[article.id],shell=document.createElement("article"),button=document.createElement("button"),source=document.createElement("a");
      shell.className="article-card-shell"; button.className="article-card"; button.type="button"; button.setAttribute("aria-pressed","false");
      button.setAttribute("aria-label","Show corpus statistics for "+article.title);
      button.innerHTML='<span class="article-card-inner"><span class="article-face article-card-front"><img src="'+esc(article.image_url)+'" alt="'+esc(article.title)+'" loading="lazy"><span class="article-copy"><small>'+article.id+' · BBC News</small><span class="article-title">'+esc(article.title)+'</span><span class="article-flip-prompt">Flip for corpus data ↻</span></span></span><span class="article-face article-card-back"><span class="article-back-heading"><small>'+article.id+' · Before and after preprocessing</small><span>Corpus statistics</span></span><span class="article-back-metrics"><span><strong>'+fmt(article.tokens.length)+'</strong><small>raw tokens</small></span><span><strong>'+fmt(article.vocabulary_size)+'</strong><small>raw vocabulary</small></span><span><strong>'+fmt(item.token_count)+'</strong><small>normalized tokens</small></span><span><strong>'+fmt(item.vocabulary_size)+'</strong><small>normalized vocabulary</small></span></span><span class="article-return-prompt">Flip back ↻</span></span></span>';
      button.addEventListener("click",function(){toggleArticle(index);});
      source.className="article-source-link"; source.href=article.url; source.target="_blank"; source.rel="noopener noreferrer"; source.textContent="Read on BBC ↗"; source.tabIndex=-1;
      shell.appendChild(button); shell.appendChild(source); rail.appendChild(shell);
    });
  }

  function bind() {
    document.querySelectorAll(".section-toggle").forEach(function(button){button.addEventListener("click",function(){
      var tasks=document.querySelectorAll("#section-"+button.dataset.section+" .task"),open=Array.prototype.some.call(tasks,function(task){return !task.open;});
      tasks.forEach(function(task){task.open=open;}); button.textContent=open?"Collapse all":"Expand all";
    });});
    document.getElementById("wordnet-metric").addEventListener("change",function(event){
      var result=state.data.section_b.B7; document.getElementById("wordnet-map").innerHTML=heatmap(result.labels,result[event.target.value]);
    });
    document.getElementById("embedding-model").addEventListener("change",function(event){
      var result=state.data.section_b; document.getElementById("embedding-map").innerHTML=heatmap(result.B7.labels,result.B8[event.target.value].similarity);
    });
  }
  function finish() {
    var a=state.data.section_a,comparisons=state.data.section_b.B9.correlations.slice();
    document.getElementById("stat-documents").textContent=a.A1.length;
    document.getElementById("stat-tokens").textContent=fmt(a.A1.reduce(function(sum,item){return sum+item.tokens.length;},0));
    document.getElementById("stat-features").textContent=fmt(a.A3.features.length);
    comparisons.sort(function(x,y){return Math.abs(y.spearman_rho)-Math.abs(x.spearman_rho);});
    var best=comparisons[0];
    document.getElementById("conclusion-grid").innerHTML='<article class="finding"><strong>Sense first.</strong><p>Explicit senses are necessary because WordNet scores depend on them.</p></article><article class="finding"><strong>Expansion depends on coverage.</strong><p>Added synonyms help only when they occur in the corpus vocabulary.</p></article><article class="finding"><strong>ρ = '+fmt(best.spearman_rho,3)+'</strong><p>The strongest observed rank agreement is '+esc(best.wordnet+' × '+best.embedding)+'.</p></article>';
  }

  fetch("data/lab2_results.json?schema=20260914").then(function(response){if(!response.ok) throw new Error("HTTP "+response.status); return response.json();})
    .then(function(data){state.data=data; state.articles=data.section_a.A1; renderArticles(); renderTasks(); bind(); finish();})
    .catch(function(error){document.getElementById("content").insertAdjacentHTML("afterbegin",'<p class="error">The experiment data could not be loaded: '+esc(error.message)+'.</p>');});
}());
