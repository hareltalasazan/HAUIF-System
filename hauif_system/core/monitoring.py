cat > hauif_system/core/monitoring.py << 'EOL'
from prometheus_client import Counter, Histogram
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter

trace.set_tracer_provider(TracerProvider())
trace.get_tracer_provider().add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))
tracer = trace.get_tracer(__name__)

REQUESTS = Counter("hauif_requests_total", "Total requests processed", ["status"])
PROCESSING_TIME = Histogram("hauif_processing_seconds", "Time spent processing cases")
CORRUPTION_CASES = Counter("hauif_corruption_cases_total", "Total corruption cases processed")
AVG_CORRUPTION_SCORE = Histogram("hauif_corruption_score", "Distribution of corruption scores")
EOL
