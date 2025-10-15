import os, json, copy, yaml, pytest, jsonschema

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
FLOW = os.path.join(ROOT, "flows", "examples", "multishot_smoke.yaml")
FLOW_SCHEMA = os.path.join(ROOT, "codex", "specs", "dsl", "v1", "flow.schema.json")

@pytest.mark.parametrize("path", [FLOW, FLOW_SCHEMA])
def test_files_exist(path):
    assert os.path.exists(path), f"Missing required file: {path}"

def test_flow_schema_strict_accepts_valid():
    with open(FLOW, "r") as f:
        data = yaml.safe_load(f)
    with open(FLOW_SCHEMA, "r") as f:
        schema = json.load(f)
    jsonschema.validate(instance=data, schema=schema)

def test_flow_schema_strict_rejects_unknown_fields():
    with open(FLOW, "r") as f:
        data = yaml.safe_load(f)
    data2 = copy.deepcopy(data)
    data2["unknown_field"] = True
    with open(FLOW_SCHEMA, "r") as f:
        schema = json.load(f)
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(instance=data2, schema=schema)

    data3 = copy.deepcopy(data)
    data3["nodes"][0]["mystery"] = 123
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(instance=data3, schema=schema)
