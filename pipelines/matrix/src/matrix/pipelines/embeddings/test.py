from pypher import __ as cypher, Pypher

from pypher.builder import FuncRaw, create_function

from pypher_utils import create_stringified_function


create_stringified_function('iterate', {'name': 'apoc.periodic.iterate'})
create_function('openai_embedding', {'name': 'apoc.ml.openai.embedding'}, func_raw=True)
create_function('set_property', {'name': 'apoc.create.setProperty'}, func_raw=True)

# class ApocIterate(MyFunc):
#     _CAPATILIZE = False # will make the resulting name all caps. Defaults to False
#     _ADD_PRECEEDING_WS = True # add whitespace before the resulting Cypher string. Defaults to True
#     _CLEAR_PRECEEDING_WS = True # add whitespace after the resulting Cypher string. Defaults to False
#     _ALIASES = ['iterate'] # aliases for your custom function. Will throw an exception if it is already defined
#     name = 'apoc.periodic.iterate' # the string that will be printed in the resulting Cypher. If this isn't defined, the class name will be used

# Build query
p = Pypher()

p.CALL.iterate(
    # Match every :Entity node in the graph
    cypher.MATCH.node('p', labels='Entity').RETURN.p,
    # For each batch, execute following statements
    [   
        # Match OpenAI embedding in a batched manner
        cypher.CALL.openai_embedding("[item in $_batch | item.p.category]", '$apiKey', "{endpoint: $endpoint, model: $model}").YIELD('index', 'text', 'embedding'),
        # Set the attribute property of the node to the embedding
        cypher.CALL.set_property('$_batch[index].p', '$attribute', 'embedding').YIELD('node').RETURN('node')
    ],
    cypher.map(
        batchMode="BATCH_SINGLE",
        batchSize=1000,
        params=cypher.map(apiKey='key', endpoint='endpoint', attribute='embedding', model="ada")
    )
).YIELD("batch", "operations")

print(str(p))