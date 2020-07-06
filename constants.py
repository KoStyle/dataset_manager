CLASS_SVR = 'svr'
CLASS_SOCAL = 'socal'
CLASS_NOCLASS = 'N/A'
RESSET = 'results'
REVSET = 'reviews'
TAGSET = 'universal'
TAG_NOUN = 'NOUN'
TAG_ADJ = 'ADJ'
TAG_VERB = 'VERB'
TAG_DET = 'DET'
TAG_PRON = 'PRON'

SEPARATOR = "\t"

TAG_RID = "rev_id"
TAG_UID = "user_id"
TAG_PID = "product_id"
TAG_UR = "user_rating"
TAG_SVR = "svr"
TAG_SOCAL = "socal"
TAG_REVIEW = "review"
TAG_CLASS = "class"

#Attr types
TYPE_STR = "STRING"
TYPE_LST = "LIST"
TYPE_NUM = "INT"

# DB CONSTANTS
DBT_CONCATS = "CONCATS"
DBT_ATTGEN = "ATTGEN"
DBT_MATTR = "MATTR"     #master attributes
DBT_MUSR = "MUSR"       #master user
DBT_MREVS = "MREVS"     #Master Reviews

MUSR_UID = "uid"
MUSR_CLASS = "class"
MUSR_DS = "dataset"
MUSR_MAEP_SVR = "maep_svr"
MUSR_MAEP_SOCAL = "maep_socal"

MREVS_RID = "rid"
MREVS_DS = "dataset"
MREVS_UID = "uid"
MREVS_PID = "pid"
MREVS_REVIEW = "review"
MREVS_SOCAL = "socal"
MREVS_SVR = "svr"

MATTR_AID = "aid"
MATTR_DESC = "desc"
MATTR_TYPE = "type"
MATTR_ACTI = "active"

ATTGEN_TID = "tid"
ATTGEN_AID = "aid"
ATTGEN_ASEQ = "aseq"  #Sequential of the attribute (for multivalue attributes or vectors (wink wink BERT)
ATTGEN_VAL = "value"
ATTGEN_CDAT = "cdate"
ATTGEN_UDAT = "udate"
ATTGEN_VER = "version"

CONCATS_TID = "tid"
CONCATS_UID = "uid"
CONCATS_NUMRE = "numrevs"
CONCATS_REVST = "revstr"


RUTA_BASE = 'ficheros_entrada/'