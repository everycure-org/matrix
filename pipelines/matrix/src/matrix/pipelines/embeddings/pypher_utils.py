
from pypher import Pypher
from pypher.builder import FuncRaw
from pypher.partial import Partial
from pypher.builder import _MODULE

class FuncWithStringifiedCypherArguments(FuncRaw):
    _CAPITALIZE = False

    def get_args(self):
        args = []

        for arg in self.args:

            # NOTE: Allows specifying multiple statements as an array
            if isinstance(arg, list):
                arg = " ".join([str(el) for el in arg])

            if isinstance(arg, (Pypher, Partial)):
                arg.parent = self.parent

            args.append(f"'{str(arg)}'")

        return ', '.join(args)
    

def create_stringified_function(name, attrs=None, func_raw=False):
    """
    This is a utility function that is used to dynamically create new
    Func or FuncRaw objects.

    Custom functions can be created and then used in Pypher:

        create_function('MyCustomFunction', {'name': 'MY_CUSTOM_FUNCTION'})
        p = Pypher()
        p.MyCustomFunction('one', 2, 'C')
        str(p) # MY_CUSTOM_FUNCTION($_PY_1123_1, $_PY_1123_2, $_PY_1123_3)

    :param str name: the name of the Func object that will be created. This
        value is used when the Pypher instance is converted to a string
    :param dict attrs: any attributes that are passed into the Func constructor
        options include `name` which will override the name param and will be
        used when the Pypher instance is converted to a string
    :param func_raw bool: A flag stating if a FuncRaw instance should be crated
        instead of a Func instance
    :return None
    """

    setattr(_MODULE, name, type(name, (FuncWithStringifiedCypherArguments,), attrs))