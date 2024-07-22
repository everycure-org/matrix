"""Custom Pypher utilities."""
from pypher import Pypher
from pypher.builder import FuncRaw
from pypher.partial import Partial
from pypher.builder import _MODULE


class Stringify(FuncRaw):
    """Pypher Stringify function.

    Custom Pypher function to represent stringification of a Cypher query. This is relevant
    for operations such as `apoc.periodic.iterate`, which expects stringified cypher queries
    as arguments.
    """

    def get_args(self):
        """Function to retrieve args."""
        args = []

        for arg in self.args:
            # NOTE: Allows specifying multiple statements as an array
            if isinstance(arg, list):
                arg = " ".join([str(el) for el in arg])

            if isinstance(arg, (Pypher, Partial)):
                arg.parent = self.parent

            args.append(f"'{arg}'")

        return ", ".join(args)

    def __unicode__(self):
        """Unicode function."""
        return self.get_args()


def create_custom_function(name, func, attrs=None, func_raw=False):
    """Create custom function.

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
    setattr(_MODULE, name, type(name, (func,), attrs))


create_custom_function("stringify", Stringify, {"name": "stringify"})
