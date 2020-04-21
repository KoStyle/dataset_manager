from io_management import read_partial_set

RUTA_BASE = 'ficheros_entrada/'

if __name__ == "__main__":
    entries = read_partial_set(RUTA_BASE + 'result-APP-SOCAL.txt')
    print(len(entries))
