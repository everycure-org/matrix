from pypher import __, Pypher

from pypher.builder import Func


from pypher import __, create_function, Pypher

from pypher.builder import Func, FuncRaw


class ApocIterate(FuncRaw):
    _CAPITALIZE = False
    _ALIASES = ["periodic_iterate", "apoc_periodic_iterate"]
    name = "apoc.periodic.iterate"


class OpenAIEmbedding(FuncRaw):
    _CAPITALIZE = False
    _ALIASES = ["openai_embedding", "apoc_ml_openai_embedding"]
    name = "apoc.ml.openai.embedding"


class ApocSetProperty(FuncRaw):
    _CAPITALIZE = False
    _ALIASES = ["set_property", "apoc_create_set_property"]
    name = "apoc.create.setProperty"


from pypher import Pypher

p = Pypher()

# p.ApocIterate(
#   f"'{__.MATCH.node('n', labels='Entity').RETURN.n}'",
#   f"'{__.openai_embedding(__.n.property('category'), '$apiKey', '$configuration').YIELD('index', 'text', 'embedding').append(__.CALL.set_property('$attr', 'embedding').YIELD.node)}'",
#   '{batchMode: "BATCH_SINGLE", batchSize: $batchSize, params: {apiKey: $apiKey, configuration: $configuration}}'
# ).YIELD('batch', 'operations')

batch_size = p.bind_param(1000, name="batchSize")
key = p.bind_param("secret", name="key")
endpoint = p.bind_param("localhost", name="endpoint")
attribute = p.bind_param("embedding", name="attribute")

p.ApocIterate(
    f"'{__.MATCH.node('n', labels='Entity').RETURN.n}'",
    f"'{__.openai_embedding(__.n.property('category'), '$apiKey', '$configuration').YIELD('index', 'text', 'embedding').append(__.CALL.set_property('$attr', 'embedding').YIELD.node.RETURN.node)}'",
    __.map(
        batchSize=batch_size,
        configuration=__.map(apiKey=key, endpoint=endpoint),
        attribute=attribute,
    ),
).YIELD("batch", "operations")

print(str(p))
print(p.bound_params)
