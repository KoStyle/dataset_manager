SELECT tid, uid, numrevs, revstr FROM CONCATS WHERE tid in (1, 3, 5, 882)

select *, case when maep_socal < maep_svr then 1 else 0 end as socal from MUSR

select * from MREVS

select * from MATTR
select * from CONCATS
SELECT C.tid, C.uid, C.numrevs, C.revstr, U.dataset FROM CONCATS C INNER JOIN MUSR U ON C.uid=U.uid order by U.uid
select * from ATTGEN where aid = 'ADJ_COUNT'
select C.uid, AVG(CAST(a.value as decimal)) as inst from ATTGEN A inner join CONCATS C on A.tid = C.tid where aid = 'PANDORA_AGE' group by c.uid order by inst
select * from ATTGEN where aid like "PAND%" order by tid
select uid, group_concat(avg, '@') as input_values, socal as output_values
from (select uid, aseq, avg, ((max + min) - (min + min)) as gap, max, min, socal
      from (
               select C.uid, a.aseq, avg(CAST(a.value as decimal)) as avg, MAX(CAST(a.value as decimal)) as max, MIN(CAST(a.value as decimal)) as min, case when maep_socal < maep_svr then '1' else '0' end as socal
               from ATTGEN A
                        inner join CONCATS C on A.tid = C.tid
                        inner join MUSR M on C.uid = M.uid
               where aid LIKE 'PAND%' and aid NOT LIKE '%PROB'
               group by c.uid, a.aid, a.aseq, socal
               order by c.uid, a.aid, a.aseq) X) XX group by uid, socal --order by CAST(uid as integer)

select * from ATTGEN a inner join CONCATS c on a.tid = c.tid where uid=102816 and aseq=0 and aid = 'BERT_VECT' order by CAST(a.value as decimal)

select * from ATTGEN where aid like '%PRO%'



DELETE from MUSR
DELETE from MREVS

DELETE from MATTR
DELETE FROM CONCATS
DELETE from ATTGEN


select c.tid, a.aid, count(*) from ATTGEN A inner join CONCATS C on A.tid = C.tid group by c.tid, A.aid



--Tetst query for 1st setup
select uid, tid, group_concat(avg, '@') as input_values, socal as output_values
from (select uid, avg, socal, svr, tid
      from (
               select C.uid, c.tid, a.aid, CAST(a.value as decimal) as avg, case when maep_socal < maep_svr then '1' else '0' end as socal, case when maep_socal >= maep_svr then '1' else '0' end as svr
               from ATTGEN A
                        inner join CONCATS C on A.tid = C.tid
                        inner join MUSR M on C.uid = M.uid
               where aid LIKE 'PAND%'
                 and aid LIKE '%PROB'
                and m.dataset='IMDB'
               order by c.uid, a.aid, a.aseq) X) XX
group by uid, socal, tid

select count(*) from ATTGEN group by tid, aid
--query for bert (oh god)
select uid, tid, group_concat(avg, '@') as input_values, socal as output_values
from (select uid, avg, socal, tid
      from (
               select C.uid, c.tid, CAST(a.value as decimal) as avg, case when maep_socal < maep_svr then '1' else '0' end as socal
               from ATTGEN A
                        inner join CONCATS C on A.tid = C.tid
                        inner join MUSR M on C.uid = M.uid
               where aid LIKE 'BERT%'
               and m.dataset='IMDB'
               order by c.uid, c.tid, a.aid, a.aseq) X) XX
group by uid, tid, socal

select * from NNRESULTS
delete from NNRESULTS where id_setup > 0
delete from NNRESULTEVO where id_setup > 0
delete from RESULT_CLASSIFICATIONS where id_setup > 0
delete from RESULT_WEIGHTS where id_setup > 0
select * from NNRESULTEVO where id_result= 30
select * from NNSETUPS
select id_result, expected, count(expected) as total, sum(case when expected = predicted then 1 else 0 end) as matching from RESULT_CLASSIFICATIONS group by id_result, expected


INSERT INTO NNSETUPS (id_setup, descrip, method, input_l_size, output_l_size, hidden_l_size, hidden_layers, ga_pairs, ga_generations, ga_crossover, ga_mutcte, ga_mutfact, ga_generr, select_statement, bp_lfact) VALUES (2, 'Desdepy2', 'BP', 6, 1, 4, 1, 40, 700, 0.8, 0.35, 0.8, 0.01, 'select uid, tid, group_concat(avg, ''@'') as input_values, socal as output_values
      from (select uid, avg, socal, tid
            from (
                     select C.uid, c.tid, CAST(a.value as decimal) as avg, case when maep_socal < maep_svr then ''1'' else ''0'' end as socal
                     from ATTGEN A
                              inner join CONCATS C on A.tid = C.tid
                              inner join MUSR M on C.uid = M.uid
                     where aid LIKE ''PAND%''
                       and aid NOT LIKE ''%PROB''
                     order by c.uid, a.aid, a.aseq) X) XX
      group by uid, socal, tid', 0.28)