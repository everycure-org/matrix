"""This module defines the Pydantic models for the BTE pipeline."""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel
import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class ConfigurableBaseModel(BaseModel):
    """Base model with configurable extra fields behavior, forbidding additional fields not explicitly defined in the schema."""

    class Config:
        """Configuration options for base model."""

        extra = "forbid"


class Log(ConfigurableBaseModel):
    """Represents a log entry containing information about the code, log level, message, and timestamp.

    Attributes:
        code (Optional[Any]): The log code, can be any type.
        level (Optional[str]): The log level (e.g., INFO, ERROR).
        message (Optional[str]): The log message describing the event.
        timestamp (Optional[str]): The timestamp of the log entry.
    """

    code: Optional[Any] = None
    level: Optional[str] = None
    message: Optional[str] = None
    timestamp: Optional[str] = None


class StatusResponse(ConfigurableBaseModel):
    """Represents the status response of a job, including job ID, status, progress, and logs.

    Attributes:
        job_id (Optional[str]): The unique identifier for the job.
        status (Optional[str]): The current status of the job.
        progress (Optional[int]): The progress of the job, represented as a percentage.
        description (Optional[str]): A human-readable description of the job.
        response_url (Optional[str]): The URL to retrieve the response.
        logs (Optional[List[Log]]): A list of log entries associated with the job.
    """

    job_id: Optional[str] = None
    status: Optional[str] = None
    progress: Optional[int] = None
    description: Optional[str] = None
    response_url: Optional[str] = None
    logs: Optional[List[Log]] = None


class Attribute(ConfigurableBaseModel):
    """Represents an attribute associated with an entity, including type, value, and metadata about the attribute.

    Attributes:
        attribute_type_id (Optional[str]): The identifier for the type of the attribute.
        value (Optional[Any]): The value of the attribute, can be any type.
        attribute_source (Optional[str]): The source of the attribute.
        value_type_id (Optional[str]): The type identifier for the value.
        value_url (Optional[str]): The URL reference for the value.
        attributes (Optional[List["Attribute"]]): A list of additional attributes.
    """

    attribute_type_id: Optional[str] = None
    value: Optional[Any] = None
    attribute_source: Optional[str] = None
    value_type_id: Optional[str] = None
    value_url: Optional[str] = None
    attributes: Optional[List["Attribute"]] = None


class Source(ConfigurableBaseModel):
    """Represents the source of information for an edge in the knowledge graph.

    Attributes:
        resource_id (Optional[str]): The unique identifier for the resource.
        resource_role (Optional[str]): The role of the resource in the context of the edge.
        upstream_resource_ids (Optional[List[str]]): A list of upstream resource identifiers.
    """

    resource_id: Optional[str] = None
    resource_role: Optional[str] = None
    upstream_resource_ids: Optional[List[str]] = None


class Node(ConfigurableBaseModel):
    """Represents a node in the knowledge or query graph, including its ID, CURIE, name, categories, and attributes.

    Attributes:
        id (Optional[str]): The unique identifier of the node.
        curie (Optional[str]): The CURIE identifier of the node.
        name (Optional[str]): The human-readable name of the node.
        categories (Optional[List[str]]): A list of categories associated with the node.
        attributes (Optional[List[Attribute]]): A list of attributes associated with the node.
        description (Optional[str]): A description of the node.
        synonym (Optional[List[str]]): A list of synonyms for the node.
        ids (Optional[List[str]]): A list of alternative identifiers for the node.
        is_set (Optional[bool]): Whether the node represents a set of entities.
        set_interpretation (Optional[str]): How the set is interpreted, if applicable.
    """

    id: Optional[str] = None
    curie: Optional[str] = None
    name: Optional[str] = None
    categories: Optional[List[str]] = None
    attributes: Optional[List[Attribute]] = None
    description: Optional[str] = None
    synonym: Optional[List[str]] = None
    ids: Optional[List[str]] = None
    is_set: Optional[bool] = None
    set_interpretation: Optional[str] = None


class Qualifier(ConfigurableBaseModel):
    """Represents a qualifier in the knowledge graph edge binding.

    Attributes:
        qualifier_type_id (str): Type identifier for the qualifier.
        qualifier_value (str): Value of the qualifier.
    """

    qualifier_type_id: str
    qualifier_value: str


class Edge(ConfigurableBaseModel):
    """Represents an edge in the knowledge or query graph, defining the relationship between two nodes.

    Attributes:
        id (Optional[str]): The unique identifier of the edge.
        subject (Optional[str]): The identifier of the subject node.
        predicate (Optional[str]): The relationship type between the subject and object.
        object (Optional[str]): The identifier of the object node.
        attributes (Optional[List[Attribute]]): A list of attributes associated with the edge.
        sources (Optional[List[Source]]): A list of sources providing the edge information.
        knowledge_type (Optional[str]): The type of knowledge the edge represents.
        provided_by (Optional[str]): The source or provider of the edge.
        publications (Optional[List[str]]): A list of publications supporting the edge.
        evidence (Optional[List[str]]): A list of evidence supporting the edge.
        qualifiers (Optional[List[str]]): Qualifiers that modify or refine the edge.
    """

    id: Optional[str] = None
    subject: Optional[str] = None
    predicate: Optional[str] = None
    object: Optional[str] = None
    attributes: Optional[List[Attribute]] = None
    sources: Optional[List[Source]] = None
    knowledge_type: Optional[str] = None
    provided_by: Optional[str] = None
    publications: Optional[List[str]] = None
    evidence: Optional[List[str]] = None
    qualifiers: Optional[List[Qualifier]] = None

    # @classmethod
    # def validate(cls, value):
    #     """Custom validation method for the Edge class to ensure that qualifiers are converted to strings if they are provided.

    #     Args:
    #         value (dict): The edge data to validate.

    #     Returns:
    #         The validated edge data.
    #     """
    #     if isinstance(value, dict):
    #         qualifiers = value.get("qualifiers")
    #         if qualifiers and isinstance(qualifiers, list):
    #             value["qualifiers"] = [str(q) for q in qualifiers]
    #     return super().validate(value)


class QueryEdge(ConfigurableBaseModel):
    """Represents an edge in the query graph used to define relationships to be queried.

    Attributes:
        knowledge_type (Optional[str]): The type of knowledge required in the query.
        object (Optional[str]): The identifier of the object node in the query graph.
        predicates (Optional[List[str]]): A list of relationship types to be queried.
        subject (Optional[str]): The identifier of the subject node in the query graph.
    """

    knowledge_type: Optional[str] = None
    object: Optional[str] = None
    predicates: Optional[List[str]] = None
    subject: Optional[str] = None


class QueryGraph(ConfigurableBaseModel):
    """Represents the query graph, containing the nodes and edges to be queried.

    Attributes:
        edges (Optional[Dict[str, QueryEdge]]): A dictionary of edges in the query graph.
        nodes (Optional[Dict[str, Node]]): A dictionary of nodes in the query graph.
    """

    edges: Optional[Dict[str, QueryEdge]] = None
    nodes: Optional[Dict[str, Node]] = None


class QueryNode(BaseModel):
    """Represents a node in the query graph used to define entities to be queried.

    Attributes:
        categories (Optional[List[str]]): A list of categories to which the node belongs.
        ids (Optional[List[str]]): A list of identifiers for the node.
        is_set (Optional[bool]): Whether the node represents a set of entities.
    """

    categories: Optional[List[str]] = None
    ids: Optional[List[str]] = None
    is_set: Optional[bool] = None


class KnowledgeGraph(ConfigurableBaseModel):
    """Represents the knowledge graph, containing the nodes and edges that describe relationships between entities.

    Attributes:
        edges (Optional[Dict[str, Edge]]): A dictionary of edges in the knowledge graph.
        nodes (Optional[Dict[str, Node]]): A dictionary of nodes in the knowledge graph.
    """

    edges: Optional[Dict[str, Edge]] = None
    nodes: Optional[Dict[str, Node]] = None


class ResultNodeBinding(ConfigurableBaseModel):
    """Represents a binding between a result node and the query node it corresponds to.

    Attributes:
        id (Optional[str]): The identifier of the result node.
        query_id (Optional[str]): The identifier of the query node.
        attributes (Optional[List[Attribute]]): A list of attributes associated with the result node.
    """

    id: Optional[str] = None
    query_id: Optional[str] = None
    attributes: Optional[List[Attribute]] = None


class ResultEdgeBinding(ConfigurableBaseModel):
    """Represents a binding between a result edge and the query edge it corresponds to.

    Attributes:
        id (Optional[str]): The identifier of the result edge.
        query_id (Optional[str]): The identifier of the query edge.
        attributes (Optional[List[Attribute]]): A list of attributes associated with the result edge.
    """

    id: Optional[str] = None
    query_id: Optional[str] = None
    attributes: Optional[List[Attribute]] = None


class Analysis(ConfigurableBaseModel):
    """Represents an analysis of the results, including edge bindings, score, and attributes.

    Attributes:
        resource_id (Optional[str]): The identifier of the resource used in the analysis.
        edge_bindings (Optional[Dict[str, List[ResultEdgeBinding]]]): A dictionary of edge bindings.
        score (Optional[float]): The score of the analysis.
        support_graphs (Optional[Any]): Any supporting graphs for the analysis.
        scoring_method (Optional[Any]): The method used to calculate the score.
        attributes (Optional[List[Attribute]]): A list of attributes associated with the analysis.
    """

    resource_id: Optional[str] = None
    edge_bindings: Optional[Dict[str, List[ResultEdgeBinding]]] = None
    score: Optional[float] = None
    support_graphs: Optional[Any] = None
    scoring_method: Optional[Any] = None
    attributes: Optional[List[Attribute]] = None


class Result(ConfigurableBaseModel):
    """Represents a result in the query response, containing node bindings and analyses.

    Attributes:
        node_bindings (Optional[Dict[str, List[ResultNodeBinding]]]): A dictionary of node bindings.
        analyses (Optional[List[Analysis]]): A list of analyses associated with the result.
        pfocr (Optional[Any]): Any additional information related to the result.
    """

    node_bindings: Optional[Dict[str, List[ResultNodeBinding]]] = None
    analyses: Optional[List[Analysis]] = None
    pfocr: Optional[Any] = None


class Message(ConfigurableBaseModel):
    """Represents the core message in a query response, containing the knowledge graph, query graph, and results.

    Attributes:
        auxiliary_graphs (Optional[Dict[str, Any]]): A dictionary of auxiliary graphs.
        knowledge_graph (Optional[KnowledgeGraph]): The knowledge graph in the response.
        query_graph (Optional[QueryGraph]): The query graph in the response.
        results (Optional[List[Result]]): A list of results in the response.
        pfocr (Optional[Any]): Any additional information related to the response.
    """

    auxiliary_graphs: Optional[Dict[str, Any]] = None
    knowledge_graph: Optional[KnowledgeGraph] = None
    query_graph: Optional[QueryGraph] = None
    results: Optional[List[Result]] = None
    pfocr: Optional[Any] = None


class InitialResponse(ConfigurableBaseModel):
    """Represents the initial response to a query, providing job details and status.

    Attributes:
        description (Optional[str]): A human-readable description of the job.
        job_id (Optional[str]): The unique identifier for the job.
        job_url (Optional[str]): The URL for retrieving the job status or result.
        status (Optional[str]): The current status of the job.
    """

    description: Optional[str] = None
    job_id: Optional[str] = None
    job_url: Optional[str] = None
    status: Optional[str] = None


class FinalResponse(ConfigurableBaseModel):
    """Represents the final response to a query, including the message, status, and logs.

    Attributes:
        message (Optional[Message]): The core message of the final response.
        biolink_version (Optional[str]): The Biolink model version used.
        description (Optional[str]): A description of the response.
        logs (Optional[List[Log]]): A list of logs generated during processing.
        schema_version (Optional[str]): The schema version used in the response.
        status (Optional[str]): The status of the final response.
        workflow (Optional[List[Dict[str, Any]]]): The workflow steps applied in processing.
    """

    message: Optional[Message] = None
    biolink_version: Optional[str] = None
    description: Optional[str] = None
    logs: Optional[List[Log]] = None
    schema_version: Optional[str] = None
    status: Optional[str] = None
    workflow: Optional[List[Dict[str, Any]]] = None


class ApiResponse(ConfigurableBaseModel):
    """Represents an API response, including metadata such as Biolink version, status, and the core message.

    Attributes:
        biolink_version (Optional[str]): The Biolink model version used.
        description (Optional[str]): A description of the response.
        logs (Optional[List[Log]]): A list of logs generated during processing.
        message (Optional[Message]): The core message of the response.
        schema_version (Optional[str]): The schema version used in the response.
        status (Optional[str]): The status of the API response.
        workflow (Optional[List[Dict[str, Any]]]): The workflow steps applied in processing.
        curie (Optional[str]): The CURIE identifier associated with the response.
        index (Optional[int]): The index of the current response in the result set.
    """

    biolink_version: Optional[str] = None
    description: Optional[str] = None
    logs: Optional[List[Log]] = None
    message: Optional[Message] = None
    schema_version: Optional[str] = None
    status: Optional[str] = None
    workflow: Optional[List[Dict[str, Any]]] = None
    curie: Optional[str] = None
    index: Optional[int] = None


class Query(ConfigurableBaseModel):
    """Represents a query sent to BioThings Explorer, containing the message, workflow, and status.

    Attributes:
        message (Optional[Message]): The core message containing the query details.
        workflow (Optional[List[Dict[str, Any]]]): The workflow steps applied in the query.
        status (Optional[str]): The current status of the query.
        description (Optional[str]): A description of the query.
        schema_version (Optional[str]): The schema version used in the query.
        logs (Optional[List[Dict[str, Any]]]): A list of logs generated during query processing.
        biolink_version (Optional[str]): The Biolink model version used in the query.
    """

    message: Optional[Message] = None
    workflow: Optional[List[Dict[str, Any]]] = None
    status: Optional[str] = None
    description: Optional[str] = None
    schema_version: Optional[str] = None
    logs: Optional[List[Dict[str, Any]]] = None
    biolink_version: Optional[str] = None
