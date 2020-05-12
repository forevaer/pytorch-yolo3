class MetricEntity(object):

    def __init__(self, name, _format=None, value=None):
        self.name = name
        self.format = _format
        self.value = value

    def setFormat(self, f):
        self.format = f

    def setValue(self, value):
        self.value = value

    def line(self):
        return self.format % self.value


metrics = [
    'name',
    'grid',
    'loss',
    'loss_x',
    'loss_y',
    'loss_w',
    'loss_h',
    'loss_confidence',
    'loss_class',
    'class_acc',
    'foreground_acc',
    'background_acc'
]

formats = {
    'grid': '%2d',
    'class_acc': '%.2f%%',
    'name': '%s',
}


class Metric(object):
    def __init__(self):
        for metric in metrics:
            _format = formats.setdefault(metric, '%0.6f')
            setattr(self, metric, MetricEntity(metric, _format))

    def setMetric(self, metric, value):
        self.metric(metric).setValue(value)

    def metric(self, metricName) -> MetricEntity:
        assert hasattr(self, metricName), f'cannot find metric : {metricName}'
        metricEntity = getattr(self, metricName)
        assert isinstance(metricEntity, MetricEntity)
        return metricEntity

    def _line(self, metric):
        assert hasattr(self, metric), f'cannot find metric : {metric}'
        metricEntity = getattr(self, metric)
        assert isinstance(metricEntity, MetricEntity)
        return metricEntity.line()

    def line(self):
        rows = []
        for metric in metrics:
            rows.append(self._line(metric))
        return rows

    @staticmethod
    def head():
        return metrics
