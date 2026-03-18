"""
Knowledge Base Integration (v0.1)

Deterministic world knowledge lookup using established knowledge bases.

Libraries:
- SPARQLWrapper: Query DBpedia/Wikidata (deterministic SPARQL)
- requests: HTTP for GeoNames API
- None of these use embeddings or ML - pure dictionary/graph lookup

Knowledge Sources:
1. DBpedia: 3M+ entities from Wikipedia infoboxes
2. Wikidata: 100M+ entities with structured properties
3. GeoNames: 11M+ geographic locations
4. ConceptNet: 8M+ common-sense assertions

Philosophy: External KBs provide facts we shouldn't hardcode
"""

from dataclasses import dataclass
from typing import List, Dict, Set, Optional, Any
from enum import Enum
import re
import time

# Libraries for KB access
try:
    from SPARQLWrapper import SPARQLWrapper, JSON
    SPARQL_AVAILABLE = True
except ImportError:
    SPARQL_AVAILABLE = False
    print("[WARN] SPARQLWrapper not installed. SPARQL queries disabled.")

import requests


# ==========================
# KNOWLEDGE BASE ENTITIES
# ==========================

@dataclass
class KnowledgeEntity:
    """
    An entity retrieved from a knowledge base.

    Example:
        entity_id: "dbr:Stanford_University"
        name: "Stanford University"
        entity_type: "University"
        properties: {"founded": 1885, "location": "California"}
    """
    entity_id: str
    name: str
    entity_type: str
    properties: Dict[str, Any]
    source: str  # 'dbpedia', 'wikidata', 'geonames', 'conceptnet'
    confidence: float


class EntityType(Enum):
    """Common entity types across KBs."""
    PERSON = "person"
    ORGANIZATION = "organization"
    LOCATION = "location"
    DATE = "date"
    NUMBER = "number"
    CONCEPT = "concept"


# ==========================
# DBPEDIA SPARQL INTERFACE
# ==========================

class DBpediaClient:
    """
    Deterministic DBpedia SPARQL queries.

    DBpedia contains 3M+ entities extracted from Wikipedia infoboxes.
    All queries are deterministic - same query always returns same results.
    """

    def __init__(self):
        if not SPARQL_AVAILABLE:
            raise ImportError("SPARQLWrapper required for DBpedia. Install: pip install SPARQLWrapper")

        self.endpoint = "http://dbpedia.org/sparql"
        self.sparql = SPARQLWrapper(self.endpoint)
        self.sparql.setReturnFormat(JSON)

        # Rate limiting (be a good citizen)
        self.min_delay = 0.5  # seconds between queries
        self.last_query_time = 0.0

    def lookup_entity(self, entity_name: str, entity_type: Optional[str] = None) -> List[KnowledgeEntity]:
        """
        Lookup an entity by name in DBpedia.

        Args:
            entity_name: Entity name (e.g., "Stanford University")
            entity_type: Optional type filter (e.g., "University")

        Returns:
            List of matching entities with properties
        """
        # Rate limiting
        elapsed = time.time() - self.last_query_time
        if elapsed < self.min_delay:
            time.sleep(self.min_delay - elapsed)

        # Build SPARQL query
        query = self._build_entity_query(entity_name, entity_type)

        try:
            self.sparql.setQuery(query)
            results = self.sparql.query().convert()
            self.last_query_time = time.time()

            return self._parse_entity_results(results)

        except Exception as e:
            print(f"[WARN] DBpedia query failed: {e}")
            return []

    def get_property(self, entity_uri: str, property_name: str) -> Optional[Any]:
        """
        Get a specific property of an entity.

        Args:
            entity_uri: DBpedia URI (e.g., "dbr:Stanford_University")
            property_name: Property name (e.g., "founded", "location")

        Returns:
            Property value or None
        """
        query = f"""
        SELECT ?value WHERE {{
            <{entity_uri}> dbo:{property_name} ?value .
        }}
        LIMIT 1
        """

        try:
            self.sparql.setQuery(query)
            results = self.sparql.query().convert()
            self.last_query_time = time.time()

            if results["results"]["bindings"]:
                return results["results"]["bindings"][0]["value"]["value"]
            return None

        except Exception as e:
            print(f"[WARN] DBpedia property query failed: {e}")
            return None

    def _build_entity_query(self, entity_name: str, entity_type: Optional[str]) -> str:
        """Build SPARQL query for entity lookup."""
        # Escape quotes in entity name
        escaped_name = entity_name.replace('"', '\\"')

        type_filter = ""
        if entity_type:
            type_filter = f"?entity a dbo:{entity_type} ."

        query = f"""
        PREFIX dbo: <http://dbpedia.org/ontology/>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

        SELECT DISTINCT ?entity ?label ?type WHERE {{
            ?entity rdfs:label ?label .
            FILTER (CONTAINS(LCASE(?label), LCASE("{escaped_name}")))
            {type_filter}
            OPTIONAL {{ ?entity a ?type }}
            FILTER (LANG(?label) = "en")
        }}
        LIMIT 10
        """
        return query

    def _parse_entity_results(self, results: Dict) -> List[KnowledgeEntity]:
        """Parse SPARQL JSON results into KnowledgeEntity objects."""
        entities = []

        for binding in results.get("results", {}).get("bindings", []):
            entity_uri = binding.get("entity", {}).get("value", "")
            label = binding.get("label", {}).get("value", "")
            entity_type = binding.get("type", {}).get("value", "")

            # Extract type name from URI
            type_name = entity_type.split("/")[-1] if entity_type else "Thing"

            entities.append(KnowledgeEntity(
                entity_id=entity_uri,
                name=label,
                entity_type=type_name,
                properties={},
                source="dbpedia",
                confidence=0.9  # High confidence for exact SPARQL match
            ))

        return entities


# ==========================
# WIKIDATA SPARQL INTERFACE
# ==========================

class WikidataClient:
    """
    Deterministic Wikidata SPARQL queries.

    Wikidata contains 100M+ entities with highly structured properties.
    More complete than DBpedia but slower queries.
    """

    def __init__(self):
        if not SPARQL_AVAILABLE:
            raise ImportError("SPARQLWrapper required for Wikidata. Install: pip install SPARQLWrapper")

        self.endpoint = "https://query.wikidata.org/sparql"
        self.sparql = SPARQLWrapper(self.endpoint)
        self.sparql.setReturnFormat(JSON)
        self.sparql.addCustomHttpHeader("User-Agent", "CHISEL/0.5 (Educational Research)")

        self.min_delay = 1.0  # Wikidata prefers slower queries
        self.last_query_time = 0.0

    def lookup_entity(self, entity_name: str, entity_type: Optional[str] = None) -> List[KnowledgeEntity]:
        """
        Lookup an entity in Wikidata.

        Args:
            entity_name: Entity name
            entity_type: Optional Wikidata class (e.g., "Q3918" for university)

        Returns:
            List of matching entities
        """
        # Rate limiting
        elapsed = time.time() - self.last_query_time
        if elapsed < self.min_delay:
            time.sleep(self.min_delay - elapsed)

        query = self._build_entity_query(entity_name, entity_type)

        try:
            self.sparql.setQuery(query)
            results = self.sparql.query().convert()
            self.last_query_time = time.time()

            return self._parse_entity_results(results)

        except Exception as e:
            print(f"[WARN] Wikidata query failed: {e}")
            return []

    def _build_entity_query(self, entity_name: str, entity_type: Optional[str]) -> str:
        """Build Wikidata SPARQL query."""
        escaped_name = entity_name.replace('"', '\\"')

        type_filter = ""
        if entity_type:
            type_filter = f"?entity wdt:P31 wd:{entity_type} ."

        query = f"""
        SELECT DISTINCT ?entity ?entityLabel WHERE {{
            ?entity rdfs:label ?label .
            FILTER (CONTAINS(LCASE(?label), LCASE("{escaped_name}")))
            {type_filter}
            FILTER (LANG(?label) = "en")
            SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
        }}
        LIMIT 10
        """
        return query

    def _parse_entity_results(self, results: Dict) -> List[KnowledgeEntity]:
        """Parse Wikidata results."""
        entities = []

        for binding in results.get("results", {}).get("bindings", []):
            entity_uri = binding.get("entity", {}).get("value", "")
            label = binding.get("entityLabel", {}).get("value", "")

            entities.append(KnowledgeEntity(
                entity_id=entity_uri,
                name=label,
                entity_type="Thing",
                properties={},
                source="wikidata",
                confidence=0.9
            ))

        return entities


# ==========================
# GEONAMES GAZETTEER
# ==========================

class GeoNamesClient:
    """
    Deterministic geographic entity lookup using GeoNames.

    GeoNames provides 11M+ locations with coordinates, populations, etc.
    Free tier: 20,000 requests/day with registration.

    To use: Register at http://www.geonames.org/login and set username.
    """

    def __init__(self, username: Optional[str] = "demo"):
        """
        Initialize GeoNames client.

        Args:
            username: GeoNames username (register at geonames.org)
        """
        self.username = username
        self.base_url = "http://api.geonames.org"

        if username == "demo":
            print("[WARN] Using demo GeoNames account. Register for full access.")

    def lookup_location(self, location_name: str, max_results: int = 10) -> List[KnowledgeEntity]:
        """
        Lookup a geographic location.

        Args:
            location_name: Location name (e.g., "Stanford", "California")
            max_results: Maximum number of results

        Returns:
            List of matching locations with coordinates and metadata
        """
        url = f"{self.base_url}/searchJSON"
        params = {
            "q": location_name,
            "maxRows": max_results,
            "username": self.username,
            "featureClass": "P",  # Populated places
            "orderby": "relevance"
        }

        try:
            response = requests.get(url, params=params, timeout=5)
            response.raise_for_status()
            data = response.json()

            return self._parse_geonames_results(data)

        except Exception as e:
            print(f"[WARN] GeoNames query failed: {e}")
            return []

    def get_location_details(self, geoname_id: int) -> Optional[KnowledgeEntity]:
        """
        Get detailed information about a location.

        Args:
            geoname_id: GeoNames ID

        Returns:
            Location entity with full details
        """
        url = f"{self.base_url}/getJSON"
        params = {
            "geonameId": geoname_id,
            "username": self.username
        }

        try:
            response = requests.get(url, params=params, timeout=5)
            response.raise_for_status()
            data = response.json()

            return self._parse_geoname_detail(data)

        except Exception as e:
            print(f"[WARN] GeoNames detail query failed: {e}")
            return None

    def _parse_geonames_results(self, data: Dict) -> List[KnowledgeEntity]:
        """Parse GeoNames search results."""
        entities = []

        for item in data.get("geonames", []):
            properties = {
                "latitude": item.get("lat"),
                "longitude": item.get("lng"),
                "population": item.get("population"),
                "country": item.get("countryName"),
                "admin1": item.get("adminName1"),  # State/province
                "feature_code": item.get("fcode")
            }

            entities.append(KnowledgeEntity(
                entity_id=f"geonames:{item.get('geonameId')}",
                name=item.get("name", ""),
                entity_type="Location",
                properties=properties,
                source="geonames",
                confidence=0.85
            ))

        return entities

    def _parse_geoname_detail(self, data: Dict) -> KnowledgeEntity:
        """Parse GeoNames detail response."""
        properties = {
            "latitude": data.get("lat"),
            "longitude": data.get("lng"),
            "population": data.get("population"),
            "country": data.get("countryName"),
            "admin1": data.get("adminName1"),
            "timezone": data.get("timezone", {}).get("timeZoneId"),
            "elevation": data.get("elevation")
        }

        return KnowledgeEntity(
            entity_id=f"geonames:{data.get('geonameId')}",
            name=data.get("name", ""),
            entity_type="Location",
            properties=properties,
            source="geonames",
            confidence=0.95
        )


# ==========================
# UNIFIED KNOWLEDGE BASE
# ==========================

class KnowledgeBase:
    """
    Unified interface to multiple knowledge bases.

    Automatically tries multiple sources and returns best results.
    All queries are deterministic - no embeddings or ML.
    """

    def __init__(self, use_dbpedia: bool = True, use_wikidata: bool = False,
                 use_geonames: bool = True, geonames_username: str = "demo"):
        """
        Initialize knowledge base with selected sources.

        Args:
            use_dbpedia: Enable DBpedia (fast, 3M entities)
            use_wikidata: Enable Wikidata (slow, 100M entities)
            use_geonames: Enable GeoNames (geographic only)
            geonames_username: GeoNames API username
        """
        self.clients = {}

        if use_dbpedia and SPARQL_AVAILABLE:
            try:
                self.clients['dbpedia'] = DBpediaClient()
                print("[OK] DBpedia client initialized")
            except Exception as e:
                print(f"[WARN] DBpedia initialization failed: {e}")

        if use_wikidata and SPARQL_AVAILABLE:
            try:
                self.clients['wikidata'] = WikidataClient()
                print("[OK] Wikidata client initialized")
            except Exception as e:
                print(f"[WARN] Wikidata initialization failed: {e}")

        if use_geonames:
            try:
                self.clients['geonames'] = GeoNamesClient(geonames_username)
                print("[OK] GeoNames client initialized")
            except Exception as e:
                print(f"[WARN] GeoNames initialization failed: {e}")

    def lookup_entity(self, entity_name: str, entity_type: Optional[EntityType] = None) -> List[KnowledgeEntity]:
        """
        Lookup an entity across all available knowledge bases.

        Args:
            entity_name: Entity name to search
            entity_type: Optional entity type filter

        Returns:
            Combined results from all sources, sorted by confidence
        """
        all_entities = []

        # Geographic entities -> GeoNames first
        if entity_type == EntityType.LOCATION and 'geonames' in self.clients:
            entities = self.clients['geonames'].lookup_location(entity_name)
            all_entities.extend(entities)

        # General entities -> DBpedia/Wikidata
        if 'dbpedia' in self.clients:
            entities = self.clients['dbpedia'].lookup_entity(entity_name)
            all_entities.extend(entities)

        if 'wikidata' in self.clients:
            entities = self.clients['wikidata'].lookup_entity(entity_name)
            all_entities.extend(entities)

        # Sort by confidence, remove duplicates
        all_entities.sort(key=lambda e: e.confidence, reverse=True)

        return self._deduplicate_entities(all_entities)

    def _deduplicate_entities(self, entities: List[KnowledgeEntity]) -> List[KnowledgeEntity]:
        """Remove duplicate entities based on name similarity."""
        if not entities:
            return []

        unique = [entities[0]]

        for entity in entities[1:]:
            # Check if this entity is similar to any already added
            is_duplicate = False
            for existing in unique:
                if self._are_similar_entities(entity, existing):
                    is_duplicate = True
                    break

            if not is_duplicate:
                unique.append(entity)

        return unique

    def _are_similar_entities(self, e1: KnowledgeEntity, e2: KnowledgeEntity) -> bool:
        """Check if two entities refer to the same thing."""
        # Same source and ID
        if e1.source == e2.source and e1.entity_id == e2.entity_id:
            return True

        # Very similar names
        from rapidfuzz import fuzz
        similarity = fuzz.ratio(e1.name.lower(), e2.name.lower())
        if similarity > 90:
            return True

        return False


# ==========================
# TESTING
# ==========================

if __name__ == "__main__":
    print("Knowledge Base Integration v0.1")
    print("=" * 60)

    # Test GeoNames (doesn't require SPARQL)
    print("\nTesting GeoNames:")
    geo = GeoNamesClient(username="demo")
    locations = geo.lookup_location("Stanford")
    for loc in locations[:3]:
        print(f"  {loc.name} ({loc.properties.get('country')})")
        print(f"    Coords: {loc.properties.get('latitude')}, {loc.properties.get('longitude')}")
        print(f"    Population: {loc.properties.get('population')}")

    # Test unified KB
    print("\nTesting Unified Knowledge Base:")
    kb = KnowledgeBase(use_dbpedia=False, use_wikidata=False, use_geonames=True)

    test_queries = ["California", "New York", "London"]
    for query in test_queries:
        print(f"\n  Query: {query}")
        results = kb.lookup_entity(query, EntityType.LOCATION)
        for result in results[:2]:
            print(f"    - {result.name} ({result.source}, confidence={result.confidence})")

    print("\n[OK] Knowledge base integration ready")
    print("[NOTE] Install SPARQLWrapper for DBpedia/Wikidata: pip install SPARQLWrapper")
