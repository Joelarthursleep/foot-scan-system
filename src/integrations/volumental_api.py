"""
Volumental API Integration
Connects to Volumental cloud services for direct scan retrieval and processing
"""

import requests
import json
import time
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
import logging
import tempfile
import shutil
from datetime import datetime, timedelta
import numpy as np

logger = logging.getLogger(__name__)

@dataclass
class VolumentalScan:
    """Volumental scan data structure"""
    scan_id: str
    customer_id: str
    scan_date: str
    store_id: str
    device_id: str

    # URLs for data access
    obj_url: str
    json_url: str
    thumbnail_url: Optional[str]

    # Measurements
    measurements: Dict

    # Quality metrics
    quality_score: float
    scan_status: str  # 'complete', 'processing', 'failed'

    # Metadata
    foot_side: str  # 'left', 'right', 'both'
    scan_type: str  # 'static', 'dynamic'
    metadata: Dict

class VolumentalAPI:
    """Client for Volumental API"""

    def __init__(self, api_key: str, api_secret: str,
                 base_url: str = "https://api.volumental.com/v2"):
        """
        Initialize Volumental API client

        Args:
            api_key: API key from Volumental
            api_secret: API secret from Volumental
            base_url: Base URL for API endpoints
        """
        self.api_key = api_key
        self.api_secret = api_secret
        self.base_url = base_url

        # Authentication headers
        self.headers = {
            'X-API-Key': api_key,
            'X-API-Secret': api_secret,
            'Content-Type': 'application/json'
        }

        # Session for connection pooling
        self.session = requests.Session()
        self.session.headers.update(self.headers)

        # Cache settings
        self.cache_dir = Path("data/volumental_cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Rate limiting
        self.last_request_time = 0
        self.min_request_interval = 0.1  # 100ms between requests

        # Verify connection
        self._verify_connection()

    def _verify_connection(self):
        """Verify API connection and credentials"""
        try:
            response = self.session.get(f"{self.base_url}/ping")
            if response.status_code == 200:
                logger.info("Successfully connected to Volumental API")
            else:
                logger.error(f"Failed to connect: {response.status_code}")
        except Exception as e:
            logger.error(f"Connection error: {e}")

    def _rate_limit(self):
        """Implement rate limiting"""
        elapsed = time.time() - self.last_request_time
        if elapsed < self.min_request_interval:
            time.sleep(self.min_request_interval - elapsed)
        self.last_request_time = time.time()

    def get_scan(self, scan_id: str) -> Optional[VolumentalScan]:
        """
        Retrieve a specific scan by ID

        Args:
            scan_id: Unique scan identifier

        Returns:
            VolumentalScan object or None if not found
        """
        self._rate_limit()

        try:
            response = self.session.get(f"{self.base_url}/scans/{scan_id}")

            if response.status_code == 200:
                data = response.json()
                return self._parse_scan_data(data)
            elif response.status_code == 404:
                logger.warning(f"Scan {scan_id} not found")
                return None
            else:
                logger.error(f"Error retrieving scan: {response.status_code}")
                return None

        except Exception as e:
            logger.error(f"Failed to get scan {scan_id}: {e}")
            return None

    def list_scans(self,
                   store_id: Optional[str] = None,
                   start_date: Optional[datetime] = None,
                   end_date: Optional[datetime] = None,
                   limit: int = 100) -> List[VolumentalScan]:
        """
        List scans with optional filters

        Args:
            store_id: Filter by store
            start_date: Filter scans after this date
            end_date: Filter scans before this date
            limit: Maximum number of scans to return

        Returns:
            List of VolumentalScan objects
        """
        self._rate_limit()

        params = {'limit': limit}

        if store_id:
            params['store_id'] = store_id
        if start_date:
            params['start_date'] = start_date.isoformat()
        if end_date:
            params['end_date'] = end_date.isoformat()

        try:
            response = self.session.get(f"{self.base_url}/scans", params=params)

            if response.status_code == 200:
                data = response.json()
                scans = [self._parse_scan_data(scan) for scan in data['scans']]
                return [s for s in scans if s is not None]
            else:
                logger.error(f"Failed to list scans: {response.status_code}")
                return []

        except Exception as e:
            logger.error(f"Error listing scans: {e}")
            return []

    def download_scan_files(self, scan: VolumentalScan,
                          output_dir: Optional[str] = None) -> Tuple[str, str]:
        """
        Download OBJ and JSON files for a scan

        Args:
            scan: VolumentalScan object
            output_dir: Directory to save files (temp dir if None)

        Returns:
            Tuple of (obj_path, json_path)
        """
        if output_dir is None:
            output_dir = tempfile.mkdtemp(prefix="volumental_")
        else:
            Path(output_dir).mkdir(parents=True, exist_ok=True)

        obj_path = Path(output_dir) / f"{scan.scan_id}.obj"
        json_path = Path(output_dir) / f"{scan.scan_id}.json"

        # Check cache first
        cached_obj = self.cache_dir / f"{scan.scan_id}.obj"
        cached_json = self.cache_dir / f"{scan.scan_id}.json"

        if cached_obj.exists() and cached_json.exists():
            logger.info(f"Using cached files for scan {scan.scan_id}")
            shutil.copy(cached_obj, obj_path)
            shutil.copy(cached_json, json_path)
            return str(obj_path), str(json_path)

        try:
            # Download OBJ file
            self._rate_limit()
            obj_response = self.session.get(scan.obj_url, stream=True)
            if obj_response.status_code == 200:
                with open(obj_path, 'wb') as f:
                    for chunk in obj_response.iter_content(chunk_size=8192):
                        f.write(chunk)
                # Cache the file
                shutil.copy(obj_path, cached_obj)
            else:
                raise Exception(f"Failed to download OBJ: {obj_response.status_code}")

            # Download JSON file
            self._rate_limit()
            json_response = self.session.get(scan.json_url)
            if json_response.status_code == 200:
                with open(json_path, 'w') as f:
                    json.dump(json_response.json(), f, indent=2)
                # Cache the file
                shutil.copy(json_path, cached_json)
            else:
                raise Exception(f"Failed to download JSON: {json_response.status_code}")

            logger.info(f"Downloaded files for scan {scan.scan_id}")
            return str(obj_path), str(json_path)

        except Exception as e:
            logger.error(f"Failed to download scan files: {e}")
            raise

    def batch_download_scans(self, scan_ids: List[str],
                           output_dir: str) -> Dict[str, Tuple[str, str]]:
        """
        Download multiple scans in batch

        Args:
            scan_ids: List of scan IDs to download
            output_dir: Directory to save files

        Returns:
            Dictionary mapping scan_id to (obj_path, json_path)
        """
        results = {}

        for scan_id in scan_ids:
            logger.info(f"Downloading scan {scan_id}...")
            scan = self.get_scan(scan_id)

            if scan:
                try:
                    paths = self.download_scan_files(scan, output_dir)
                    results[scan_id] = paths
                except Exception as e:
                    logger.error(f"Failed to download {scan_id}: {e}")
            else:
                logger.warning(f"Scan {scan_id} not found")

        return results

    def get_scan_measurements(self, scan_id: str) -> Optional[Dict]:
        """
        Get detailed measurements for a scan

        Args:
            scan_id: Scan identifier

        Returns:
            Dictionary of measurements or None
        """
        self._rate_limit()

        try:
            response = self.session.get(f"{self.base_url}/scans/{scan_id}/measurements")

            if response.status_code == 200:
                return response.json()
            else:
                logger.error(f"Failed to get measurements: {response.status_code}")
                return None

        except Exception as e:
            logger.error(f"Error getting measurements: {e}")
            return None

    def search_scans(self, query: Dict) -> List[VolumentalScan]:
        """
        Search scans with advanced query

        Args:
            query: Search query parameters
                - customer_email: Customer email
                - customer_phone: Customer phone
                - size_range: Tuple of (min_size, max_size)
                - quality_threshold: Minimum quality score
                - date_range: Tuple of (start_date, end_date)

        Returns:
            List of matching scans
        """
        self._rate_limit()

        # Build query parameters
        params = {}

        if 'customer_email' in query:
            params['customer_email'] = query['customer_email']
        if 'customer_phone' in query:
            params['customer_phone'] = query['customer_phone']
        if 'size_range' in query:
            params['size_min'] = query['size_range'][0]
            params['size_max'] = query['size_range'][1]
        if 'quality_threshold' in query:
            params['quality_min'] = query['quality_threshold']
        if 'date_range' in query:
            params['start_date'] = query['date_range'][0].isoformat()
            params['end_date'] = query['date_range'][1].isoformat()

        try:
            response = self.session.get(f"{self.base_url}/scans/search", params=params)

            if response.status_code == 200:
                data = response.json()
                scans = [self._parse_scan_data(scan) for scan in data['results']]
                return [s for s in scans if s is not None]
            else:
                logger.error(f"Search failed: {response.status_code}")
                return []

        except Exception as e:
            logger.error(f"Search error: {e}")
            return []

    def _parse_scan_data(self, data: Dict) -> Optional[VolumentalScan]:
        """Parse API response into VolumentalScan object"""
        try:
            return VolumentalScan(
                scan_id=data['scan_id'],
                customer_id=data.get('customer_id', ''),
                scan_date=data.get('scan_date', ''),
                store_id=data.get('store_id', ''),
                device_id=data.get('device_id', ''),
                obj_url=data.get('obj_url', ''),
                json_url=data.get('json_url', ''),
                thumbnail_url=data.get('thumbnail_url'),
                measurements=data.get('measurements', {}),
                quality_score=data.get('quality_score', 0.0),
                scan_status=data.get('status', 'unknown'),
                foot_side=data.get('foot_side', 'unknown'),
                scan_type=data.get('scan_type', 'static'),
                metadata=data.get('metadata', {})
            )
        except Exception as e:
            logger.error(f"Failed to parse scan data: {e}")
            return None

    def upload_analysis_results(self, scan_id: str, analysis: Dict) -> bool:
        """
        Upload analysis results back to Volumental

        Args:
            scan_id: Scan identifier
            analysis: Analysis results dictionary

        Returns:
            True if successful
        """
        self._rate_limit()

        try:
            response = self.session.post(
                f"{self.base_url}/scans/{scan_id}/analysis",
                json=analysis
            )

            if response.status_code in [200, 201]:
                logger.info(f"Successfully uploaded analysis for {scan_id}")
                return True
            else:
                logger.error(f"Failed to upload analysis: {response.status_code}")
                return False

        except Exception as e:
            logger.error(f"Error uploading analysis: {e}")
            return False

class VolumentalWebhookHandler:
    """Handle webhooks from Volumental for real-time updates"""

    def __init__(self, webhook_secret: str):
        """
        Initialize webhook handler

        Args:
            webhook_secret: Secret for webhook verification
        """
        self.webhook_secret = webhook_secret

    def verify_webhook(self, request_data: bytes, signature: str) -> bool:
        """
        Verify webhook signature

        Args:
            request_data: Raw request body
            signature: Signature from header

        Returns:
            True if signature is valid
        """
        import hmac
        import hashlib

        expected_signature = hmac.new(
            self.webhook_secret.encode(),
            request_data,
            hashlib.sha256
        ).hexdigest()

        return hmac.compare_digest(expected_signature, signature)

    def handle_scan_complete(self, data: Dict) -> Dict:
        """
        Handle scan completion webhook

        Args:
            data: Webhook payload

        Returns:
            Response data
        """
        scan_id = data.get('scan_id')
        logger.info(f"Received scan complete webhook for {scan_id}")

        # Trigger processing pipeline
        response = {
            'status': 'received',
            'scan_id': scan_id,
            'action': 'process_scan'
        }

        return response

    def handle_scan_failed(self, data: Dict) -> Dict:
        """
        Handle scan failure webhook

        Args:
            data: Webhook payload

        Returns:
            Response data
        """
        scan_id = data.get('scan_id')
        error = data.get('error', 'Unknown error')
        logger.error(f"Scan {scan_id} failed: {error}")

        response = {
            'status': 'received',
            'scan_id': scan_id,
            'action': 'log_failure'
        }

        return response

class VolumentalDataSync:
    """Synchronize Volumental data with local database"""

    def __init__(self, api_client: VolumentalAPI, local_db_path: str):
        """
        Initialize data synchronization

        Args:
            api_client: VolumentalAPI instance
            local_db_path: Path to local database
        """
        self.api_client = api_client
        self.local_db_path = Path(local_db_path)

        # Initialize local database connection
        import sqlite3
        self.conn = sqlite3.connect(str(self.local_db_path))
        self._init_sync_tables()

    def _init_sync_tables(self):
        """Initialize synchronization tables"""
        cursor = self.conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS scan_sync (
                scan_id TEXT PRIMARY KEY,
                customer_id TEXT,
                scan_date TEXT,
                last_sync TEXT,
                sync_status TEXT,
                local_path TEXT,
                metadata TEXT
            )
        """)

        self.conn.commit()

    def sync_new_scans(self, since: Optional[datetime] = None) -> int:
        """
        Sync new scans from Volumental

        Args:
            since: Sync scans created after this date

        Returns:
            Number of new scans synced
        """
        if since is None:
            # Get last sync time from database
            cursor = self.conn.cursor()
            cursor.execute("""
                SELECT MAX(last_sync) FROM scan_sync
            """)
            result = cursor.fetchone()
            if result and result[0]:
                since = datetime.fromisoformat(result[0])
            else:
                since = datetime.now() - timedelta(days=7)  # Default to last week

        # Get new scans from API
        new_scans = self.api_client.list_scans(start_date=since)

        synced_count = 0
        for scan in new_scans:
            if self._sync_scan(scan):
                synced_count += 1

        logger.info(f"Synced {synced_count} new scans")
        return synced_count

    def _sync_scan(self, scan: VolumentalScan) -> bool:
        """Sync individual scan to local database"""
        try:
            # Download scan files
            output_dir = Path("data/volumental_scans") / scan.scan_id
            output_dir.mkdir(parents=True, exist_ok=True)

            obj_path, json_path = self.api_client.download_scan_files(
                scan, str(output_dir)
            )

            # Update sync database
            cursor = self.conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO scan_sync
                (scan_id, customer_id, scan_date, last_sync, sync_status, local_path, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                scan.scan_id,
                scan.customer_id,
                scan.scan_date,
                datetime.now().isoformat(),
                'synced',
                str(output_dir),
                json.dumps(scan.metadata)
            ))

            self.conn.commit()
            return True

        except Exception as e:
            logger.error(f"Failed to sync scan {scan.scan_id}: {e}")
            return False