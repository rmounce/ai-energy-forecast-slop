"""Parse archived Python representations without evaluating arbitrary code."""
import ast
from datetime import datetime, timedelta, timezone
import json
from zoneinfo import ZoneInfo


def parse_rows(raw):
    if not isinstance(raw, str) or len(raw) > 2_000_000:
        raise ValueError('missing or oversized archived rows')
    try:
        result = json.loads(raw)
    except json.JSONDecodeError:
        tree = ast.parse(raw, mode='eval')
        if sum(1 for _ in ast.walk(tree)) > 100_000: raise ValueError('oversized archived syntax')
        def value(node):
            if isinstance(node, ast.Constant): return node.value
            if isinstance(node, (ast.List, ast.Tuple)): return [value(item) for item in node.elts]
            if isinstance(node, ast.Dict):
                keys = [value(key) for key in node.keys]
                if any(not isinstance(key, str) for key in keys) or len(set(keys)) != len(keys):
                    raise ValueError('invalid archived dictionary keys')
                return dict(zip(keys, [value(item) for item in node.values]))
            if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
                item = value(node.operand)
                if type(item) not in (int, float): raise ValueError('invalid archived signed value')
                return -item if isinstance(node.op, ast.USub) else item
            if isinstance(node, ast.Attribute) and ast.dump(node) == ast.dump(ast.parse('datetime.timezone.utc', mode='eval').body):
                return timezone.utc
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name):
                name = node.func.value.id+'.'+node.func.attr
                constructors = {'datetime.datetime': datetime, 'datetime.timedelta': timedelta,
                    'datetime.timezone': timezone, 'zoneinfo.ZoneInfo': ZoneInfo}
                if name not in constructors or any(key.arg is None for key in node.keywords):
                    raise ValueError('unsupported archived constructor')
                args, kwargs = [value(item) for item in node.args], {key.arg: value(key.value) for key in node.keywords}
                if name == 'zoneinfo.ZoneInfo':
                    zone = args[0] if len(args) == 1 else kwargs.get('key')
                    if zone not in ('UTC', 'Australia/Adelaide'): raise ValueError('unsupported archived timezone')
                return constructors[name](*args, **kwargs)
            raise ValueError('unsupported archived syntax')
        result = value(tree.body)
    def normalise(item):
        if isinstance(item, datetime):
            if item.tzinfo is None: raise ValueError('naive archived datetime')
            return item.isoformat()
        if isinstance(item, dict): return {key: normalise(item) for key, item in item.items()}
        if isinstance(item, list): return [normalise(item) for item in item]
        return item
    result = normalise(result)
    if not isinstance(result, list) or not 1 <= len(result) <= 1000 or not all(isinstance(row, dict) for row in result):
        raise ValueError('invalid archived row shape')
    return result
