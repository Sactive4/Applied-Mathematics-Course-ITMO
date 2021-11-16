import numpy as np

from lab4.table import get_fns, solve, Table, TableType

from lab5.game import Game


def solve_game(fn, debug=False):
    game = Game.load(fn)

    print("*********** ", fn, " ***********")
    print("Clean strategy solution:")
    print("OURS: ", game.get_clean_strategy())
    print("CORRECT: ", game.answer_clean)

    print("Mixed strategy solution:")
    s = game.solve_simplex(debug)
    print("OURS: ", s[0], s[1])
    print("CORRECT: ", game.answer_simplex)



def get_fns():
    from os import listdir
    from os.path import isfile, join
    return sorted(
        ["tasks/" + f for f in listdir("./tasks/") if isfile(join("./tasks/", f))]
    )


if __name__ == "__main__":

    debug = False

    for fn in get_fns():

        solve_game(fn, debug)

        # if solution is None:
        #     print("... SKIPPED ", fn)
        # elif (answer is None) or len(answer) == 0:
        #     print("? ", solution, " Add answer to evaluate. ", fn)
        # elif np.allclose(solution, answer) or abs(value1 - value2) < 0.00001:
        #     print("+ ", solution, " OK ", fn)
        # else:
        #     print("- ", solution, " WRONG ", fn)
