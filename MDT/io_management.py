

SEPARATOR=";"

TAG_ID="id"

TAG_SVR="svr"
TAG_SOCAL="socal"
TAG_TEXT= "comment"


#TODO Rework header reading
def read_og_set_file(file_name):
    entries=[]
    f = open(file_name, "r")

    buffer= f.readlines()
    data_header= buffer.pop(0)
    data_header_tokens= data_header.split(SEPARATOR)

    header_list=[]

    for data_token in data_header_tokens:
        header_list.append(data_token.strip().lower())

    for line in buffer:
        values={}
        tokens=line.split(SEPARATOR)
        index=0
        for token in tokens:
            values[header_list[index]]=token.strip().lower()
            index+=1
        entries.append(values)

    return entries


def split_set(entries):
    set1=[]
    set2=[]
    sets=[set1, set2]
    for entry in entries:
        if entry[TAG_SVR]>entry[TAGTAG_SOCAL]:
            set1.append(entry)
        else:
            set2.append(entry)

    return sets


def get_entry(id, set):

    i=0
    entry=set[i]
    max_len=len(set)
    while not entry[TAG_ID]==id and i<max_len:
        i+=1
        entry=set[i]

    if i==max_len:
        return None
    else:
        return entry

#This method joins two sets that are linked through the field id. We search for the same ID in both sets and join said entries to create a complete set
def join_partial_set_entries(set_comments, set_results):
    







