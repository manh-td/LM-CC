class DataProcessor:
    def __init__(self, data):
        self.data = data
        self.results = {}
    
    def _analyze_items(self, items):
        analyzed = {}
        for item in items:
            key = item.get('name', 'unknown')
            analyzed[key] = self._calculate_metrics(item)
        return analyzed
    
    def _calculate_metrics(self, item):
        values = item.get('values', [])
        
        if values:
            if len(values) > 5:
                metric_type = 'large'
            elif len(values) > 2:
                metric_type = 'medium'
            else:
                metric_type = 'small'
        else:
            metric_type = 'empty'
        
        metrics = {
            'sum': sum(values),
            'avg': self._get_average(values),
            'type': metric_type,
            'metadata': self._process_metadata(item.get('meta', {}))
        }
        return metrics
    
    def _process_metadata(self, meta):
        processed = {}
        for key, value in meta.items():
            if isinstance(value, list):
                if len(value) > 0:
                    if isinstance(value[0], list):
                        processed[key] = self._flatten_list(value)
                    else:
                        processed[key] = value
                else:
                    processed[key] = []
            elif isinstance(value, dict):
                processed[key] = value
            elif isinstance(value, str):
                processed[key] = value.upper()
            else:
                processed[key] = value
        return processed
    
    def _flatten_list(self, lst):
        result = []
        for item in lst:
            if isinstance(item, list):
                result.extend(self._flatten_list(item))
            else:
                result.append(item)
        return result