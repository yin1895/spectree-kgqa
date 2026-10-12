#!/usr/bin/env python3
"""Map public KGQA question formats into SpecTree's canonical JSONL schema."""
from __future__ import annotations
import argparse,json,re
from pathlib import Path
from spectree.experiments.io import write_jsonl


def seq(x):
    return x if isinstance(x,list) else []


def get_answers(r, typ):
    rows=[]
    if typ=='webqsp':
        for parse in seq(r.get('Parses')):
            rows.extend(seq(parse.get('Answers')))
    else:
        rows=seq(r.get('answers',r.get('answer',r.get('Answers',[]))))
    values=[]
    for ans in rows:
        if isinstance(ans,str): values.append(ans)
        elif isinstance(ans,dict):
            v=next((ans[k] for k in ('AnswerArgument','answer_id','answer_argument','id','mid','kb_id','entity_id','answer')
                    if k in ans and isinstance(ans[k],(str,int))), None)
            if v is not None: values.append(str(v))
    return sorted(set(values))


def links_from_row(r, typ):
    if typ=='webqsp':
        return sorted({p['TopicEntityMid'] for p in seq(r.get('Parses')) if p.get('TopicEntityMid')})
    entity=r.get('topic_entities',r.get('topic_entity',r.get('topic_entity_ids',[])))
    if isinstance(entity,str):return [entity]
    if isinstance(entity,list):return [str(x) for x in entity]
    if typ=='grailqa':
        query=r.get('graph_query') or {}
        return sorted({str(n['id']) for n in seq(query.get('nodes'))
                      if n.get('node_type')=='entity' and 'id' in n})
    return []


def convert(source, typ, split='', override=None):
    result=[]
    if typ in ('webqsp','cwq','grailqa','canonical'):
        if isinstance(source,dict) and typ=='webqsp': source=source.get('Questions',[])
        if not isinstance(source,list):raise ValueError('Expected a list of questions')
        for n,r in enumerate(source):
            qid=str(r.get('question_id',r.get('QuestionId',r.get('ID',r.get('qid',n)))))
            question=r.get('question',r.get('RawQuestion',r.get('ProcessedQuestion')))
            if not isinstance(question,str) or not question:raise ValueError(f'{qid}: missing question')
            roots=override.get(qid,links_from_row(r,typ)) if override else links_from_row(r,typ)
            gold=get_answers(r,typ)
            if typ=='canonical':gold=seq(r.get('answers',[]))
            level=r.get('level',r.get('split_level'))
            row={'question_id':qid,'question':question,'topic_entities':roots,
                 'answers':gold,'split':split or r.get('split','')}
            if level is not None:row['level']=level
            if r.get('cluster_id') is not None:row['cluster_id']=r['cluster_id']
            result.append(row)
    else: raise ValueError('Unsupported input format')
    if len(result)!=len({r['question_id'] for r in result}):raise ValueError('Duplicate question IDs')
    return result


def load_json_or_jsonl(path):
    if path.suffix=='.jsonl':return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return json.loads(path.read_text(encoding='utf8'))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--format',choices=['webqsp','cwq','grailqa','metaqa','pathquestion','canonical'],required=True)
    p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--split',default='');p.add_argument('--entity-links',type=Path)
    p.add_argument('--ids-out',type=Path)
    a=p.parse_args()
    overrides=json.loads(a.entity_links.read_text()) if a.entity_links else None
    if a.format in ('metaqa','pathquestion'):
        rows=[]
        for n,line in enumerate(a.input.read_text(encoding='utf8').splitlines()):
            if not line.strip():continue
            parts=line.split('\t')
            if len(parts)<2:raise ValueError(f'Expected tab-separated question/answer at {n+1}')
            q=parts[0].strip(); answer=parts[1].strip().split('|')
            roots=re.findall(r'\[([^\]]+)\]',q)
            qid=f'{a.format}-{n:06d}'
            rows.append({'question_id':qid,'question':q,'topic_entities':overrides.get(qid,roots) if overrides else roots,
                         'answers':sorted(set(a.strip() for a in answer if a.strip())),'split':a.split})
    else:
        rows=convert(load_json_or_jsonl(a.input),a.format,a.split,overrides)
    write_jsonl(a.output,rows)
    if a.ids_out:
        a.ids_out.parent.mkdir(parents=True,exist_ok=True)
        a.ids_out.write_text(json.dumps([r['question_id'] for r in rows],indent=2)+'\n')
    print(json.dumps({'questions':len(rows),'with_roots':sum(bool(r['topic_entities']) for r in rows),
                      'output':str(a.output)},indent=2))

if __name__=='__main__':main()
