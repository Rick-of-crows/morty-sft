import time
from typing import List, Union, Optional, Tuple, Dict, Any

from googleapiclient import discovery  # pip install google-api-python-client
from googleapiclient.errors import HttpError

##############################
# Perspective API
##############################
PERSPECTIVE_API_LEN_LIMIT = 20480

# All attributes can be found here:
# https://github.com/conversationai/perspectiveapi/blob/master/2-api/models.md
PERSPECTIVE_API_ATTRIBUTES = (
    'TOXICITY',
    'SEVERE_TOXICITY',
    'IDENTITY_ATTACK',
    'INSULT',
    'THREAT',
    'PROFANITY',
)
PERSPECTIVE_API_ATTRIBUTES_LOWER = tuple(a.lower() for a in PERSPECTIVE_API_ATTRIBUTES)
PERSPECTIVE_API_KEY = 'AIzaSyBNhNIc4rdfxeLkiyaoCr2_49ARALLlaF8'  # google cloud API key
PROXY_HOST = "10.135.24.25"
PROXY_PORT = 7890


def unpack_scores(response_json: dict) -> Optional[Tuple[dict, dict]]:
    if not response_json:
        return None

    attribute_scores = response_json['attributeScores'].items()

    summary_scores = {}
    span_scores = {}
    for attribute, scores in attribute_scores:
        attribute = attribute.lower()

        # Save summary score
        assert scores['summaryScore']['type'] == 'PROBABILITY'
        summary_scores[attribute] = scores['summaryScore']['value']

        # Save span scores
        for span_score_dict in scores['spanScores']:
            assert span_score_dict['score']['type'] == 'PROBABILITY'
            span = (span_score_dict['begin'], span_score_dict['end'])
            span_scores.setdefault(span, {})[attribute] = span_score_dict['score']['value']

    return summary_scores, span_scores


class PerspectiveAPI:
    def __init__(self, api_key: str = PERSPECTIVE_API_KEY, rate_limit: int = 25, proxy_host=None, proxy_port=None):
        self.proxy_host = proxy_host
        self.proxy_port = proxy_port
        self.service = self._make_service(api_key, proxy_host, proxy_port)
        self.last_request_time = -1  # satisfies initial condition
        self.rate_limit = rate_limit
        self.next_uid = 0

    def request(self, texts: Union[str, List[str]]) -> List[Tuple[Optional[Dict[str, Any]], Optional[HttpError]]]:
        if isinstance(texts, str):
            texts = [texts]

        # Rate limit to 1 batch request per second
        assert len(texts) <= self.rate_limit
        time_since_last_request = time.time() - self.last_request_time
        if time_since_last_request < 1:
            time.sleep(1 - time_since_last_request)
        self.last_request_time = time.time()

        # Keys guaranteed in insertion order (Python 3.7+)
        responses = {str(uid): None for uid in range(self.next_uid, self.next_uid + len(texts))}
        self.next_uid += len(texts)

        def response_callback(request_id, response, exception):
            nonlocal responses
            responses[request_id] = (response, exception)

        # Make API request
        batch_request = self.service.new_batch_http_request()
        for uid, text in zip(responses.keys(), texts):
            batch_request.add(self._make_request(text, self.service), callback=response_callback, request_id=uid)
        batch_request.execute()

        return list(responses.values())

    @staticmethod
    def _make_service(api_key: str, proxy_host=None, proxy_port=None):
        if proxy_host is not None and proxy_port is not None:
            # Generate API client object dynamically based on service name and version
            import httplib2
            proxy_types = [
                httplib2.socks.PROXY_TYPE_SOCKS5,
                httplib2.socks.PROXY_TYPE_SOCKS4,
                httplib2.socks.PROXY_TYPE_HTTP,
            ]
            proxy_types_str = ["PROXY_TYPE_HTTP", "PROXY_TYPE_SOCKS4", "PROXY_TYPE_SOCKS5"]
            service = None
            for i, proxy_type in enumerate(proxy_types):
                proxy_info = httplib2.ProxyInfo(proxy_type=proxy_type, proxy_host=PROXY_HOST,
                                                proxy_port=PROXY_PORT)
                http = httplib2.Http(timeout=10, proxy_info=proxy_info, disable_ssl_certificate_validation=False)
                try:
                    service = discovery.build('commentanalyzer', 'v1alpha1', developerKey=PERSPECTIVE_API_KEY,
                                              http=http,
                                              static_discovery=False)
                    print("find support PROXY_TYPE: {}".format(proxy_types_str[i]))
                    break
                except:
                    print("{} not support, will try another PROXY_TYPE".format(proxy_types_str[i]))
            if service is None:
                raise Exception("The Proxy Failed !!!")
        else:
            service = discovery.build('commentanalyzer', 'v1alpha1', developerKey=api_key, static_discovery=False)
        return service

    @staticmethod
    def _make_request(text: str, service):
        analyze_request = {
            'comment': {'text': text},
            'requestedAttributes': {attr: {} for attr in PERSPECTIVE_API_ATTRIBUTES},
            'spanAnnotations': True,
        }
        return service.comments().analyze(body=analyze_request)


def test_perspective_api():
    proxy_host = PROXY_HOST
    proxy_port = PROXY_PORT
    api = PerspectiveAPI(proxy_host=proxy_host, proxy_port=proxy_port)

    text_success = "Testing"
    text_error = 'x' * (20480 + 1)

    score_1, error_1 = api.request(text_success)[0]
    print("score_1: {}".format(score_1))
    summary_scores, _ = unpack_scores(score_1)
    print("summary_scores:{}".format(summary_scores))


if __name__ == '__main__':
    test_perspective_api()

