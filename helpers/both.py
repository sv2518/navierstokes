from firedrake import *

def both(expr):
    return expr('+') + expr('-')