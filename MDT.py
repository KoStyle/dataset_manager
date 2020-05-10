import nltk
import timeit

from attribute_management import generate_attributes
from constants import REVSET
from io_management import read_partial_set, assing_class, join_result_sets, join_partial_set_entries

from util import chronometer
from util import print_chrono

from att_generators.void_attr_gen import VoidAttGen

RUTA_BASE = 'ficheros_entrada/'

# TODO: una integración de textos de un UserCase
# TODO: Crear un fichero de set nuevo de user cases, una sola linea de texto, usuario, maeps o clase directamente, y review texto (el texto a parte quiza)
# TODO: usar el nuevo set para attribute_management y seguir adelante como planeado


if __name__ == "__main__":
    entries_socal_app = read_partial_set(RUTA_BASE + 'result-IMDB-SOCAL.txt')
    entries_svr_app = read_partial_set(RUTA_BASE + 'result-IMDB-SVR62.txt')
    entries_comments = read_partial_set(RUTA_BASE + 'revs_imdb.txt', REVSET)
    complete_results = assing_class(join_result_sets(entries_socal_app, entries_svr_app))
    user_cases = join_partial_set_entries(complete_results)


    # print(len(user_cases))

    @chronometer
    def test():
        user_case = list(user_cases.items())[0][1]
        texto = user_case.get_text(entries_comments)
        # print("Longitud del texto: %d" % len(texto))
        list_generators = []
        gen_tmp = VoidAttGen.gen_wc()
        gen_noun = VoidAttGen.gen_noun()
        gen_adj = VoidAttGen.gen_adj()
        list_generators.append(VoidAttGen.gen_verb())
        list_generators.append(VoidAttGen.gen_det())
        list_generators.append(VoidAttGen.gen_pron())
        list_generators.append(VoidAttGen.gen_sentences())

        list_generators.append(gen_tmp)
        list_generators.append(gen_noun)
        list_generators.append(gen_adj)

        dict_temporal = {user_case.get_id(): user_case}
        generate_attributes(dict_temporal, entries_comments, list_generators)

        print(user_case.attributes)


    test()
    print_chrono()

    # for tupla in user_cases.items():
    #     user = tupla[1]
    #
    #     print("Maeps %s %s" % (user.get_maep_socal(), user.get_maep_svr()))

# print(len(sorted_by_class[TAG_SOCAL]))

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('taggers/averaged_perceptron_tagger')
except LookupError:
    nltk.download('averaged_perceptron_tagger')

try:
    nltk.data.find('taggers/universal_tagset')
except LookupError:
    nltk.download('universal_tagset')

tokens = nltk.word_tokenize("My name is Phillip Douglass, the most beautilful in this amazing world, bitch.")

tagged = nltk.pos_tag(tokens, tagset='universal')

print(tagged)
