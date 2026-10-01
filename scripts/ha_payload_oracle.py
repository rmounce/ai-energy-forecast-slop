"""Offline oracle executed inside the deployed HA container; never calls services."""
import ast
import json
import sys
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo
from homeassistant.helpers.template import TemplateEnvironment


def run(request):
    states = request['states']
    instant = datetime.fromisoformat(request['captured_at'])
    local = ZoneInfo(request.get('timezone', 'Australia/Adelaide'))

    def as_datetime(value):
        return value if isinstance(value, datetime) else datetime.fromisoformat(value.replace('Z', '+00:00'))

    def state_attr(entity, key):
        value = states.get(entity, {}).get('attributes', {}).get(key)
        if key == 'detailedForecast' and value is not None:
            return [dict(row, period_start=as_datetime(row['period_start'])) for row in value]
        return value

    env = TemplateEnvironment(None)
    env.globals.update(states=lambda entity: states.get(entity, {}).get('state', 'unknown'),
                       state_attr=state_attr, utcnow=lambda: instant.astimezone(timezone.utc),
                       now=lambda: instant.astimezone(local), as_datetime=as_datetime,
                       as_local=lambda value: value.astimezone(local), timedelta=timedelta)
    env.filters.update(state_attr=state_attr, as_datetime=as_datetime,
                       as_local=lambda value: value.astimezone(local))

    def render(source, context):
        output = env.from_string(source).render(context).strip()
        try:
            return json.loads(output)
        except (ValueError, TypeError):
            try:
                return ast.literal_eval(output)
            except (ValueError, SyntaxError):
                return output

    result = {}
    for kind in ('dh', 'mpc'):
        context = {}
        for variables in request['templates'][kind]['variables']:
            # HA script variables may refer to earlier variables in the same block.
            for name, source in variables.items():
                context[name] = render(source, context) if isinstance(source, str) else source
        result[kind] = render(request['templates'][kind]['payload'], context)
    return result


if __name__ == '__main__':
    print(json.dumps(run(json.load(sys.stdin))))
