from spectree.core.codec import Candidate, encode, recover
from spectree.core.incremental_codec import SharedPrefixIndex


def bank():
    ctx={'graph':'example','root':'Mira','executor':'v1'}
    u={'head':'Mira','relation':'studied_at','direction':'+','tail':'University of Oxford'}
    a={'head':'University of Oxford','relation':'located_in','direction':'+','tail':'Oxford'}
    b={'head':'University of Oxford','relation':'academic_partner','direction':'+','tail':'University of Toronto'}
    c={'head':'Oxford','relation':'in_country','direction':'+','tail':'United Kingdom'}
    d={'head':'University of Toronto','relation':'in_country','direction':'+','tail':'Canada'}
    return [Candidate('c1',ctx,(u,a,c),-0.2,('United Kingdom',)),
            Candidate('c2',ctx,(u,b,d),-0.3,('Canada',))]


def test_roundtrip_three_encodings():
    src=bank()
    for fmt in ('flat','dictionary','shared'):
        packet,registry=encode(src,fmt)
        assert [c.record() for c in recover(packet,registry)]==[c.record() for c in src]
        assert len(packet['terminals'])==2
    assert len(encode(src,'shared')[0]['records'])==5


def test_incremental_trial_and_rollback():
    src=bank();index=SharedPrefixIndex()
    index.commit(src[:1]);before=index.export()
    packet,registry=index.trial(src)
    assert len(packet['terminals'])==2
    assert index.export()==before
    index.commit(src)
    assert len(index.export()[0]['records'])==5
