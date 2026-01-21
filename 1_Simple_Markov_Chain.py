import json
import requests
from typing import Dict, Any, List
from pprint import pprint


class HighchartsJSONParser:
    """Class to fetch and parse Highcharts API JSON data."""
    
    def __init__(self, url: str = "https://api.highcharts.com/highcharts/tree.json"):
        """
        Initialize the parser with the API URL.
        
        Args:
            url: The URL to fetch JSON data from
        """
        self.url = url
        self.data = None
    
    def fetch_data(self) -> Dict[str, Any]:
        """
        Fetch JSON data from the API endpoint.
        
        Returns:
            Parsed JSON data as a dictionary
            
        Raises:
            requests.RequestException: If the request fails
            json.JSONDecodeError: If the response is not valid JSON
        """
        try:
            print(f"Fetching data from {self.url}...")
            response = requests.get(self.url, timeout=30)
            response.raise_for_status()  # Raise an error for bad status codes
            
            self.data = response.json()
            print(f"Successfully fetched {len(str(self.data))} characters of JSON data")
            return self.data
            
        except requests.RequestException as e:
            print(f"Error fetching data: {e}")
            raise
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON: {e}")
            raise
    
    def get_metadata(self) -> Dict[str, Any]:
        """
        Extract metadata from the JSON response.
        
        Returns:
            Dictionary containing metadata information
        """
        if not self.data:
            self.fetch_data()
        
        return self.data.get('_meta', {})
    
    def get_all_options(self) -> List[str]:
        """
        Get a list of all top-level option keys.
        
        Returns:
            List of option names
        """
        if not self.data:
            self.fetch_data()
        
        # Exclude metadata key
        return [key for key in self.data.keys() if key != '_meta']
    
    def get_option(self, option_name: str) -> Dict[str, Any]:
        """
        Get details for a specific option.
        
        Args:
            option_name: The name of the option to retrieve
            
        Returns:
            Dictionary containing option details
        """
        if not self.data:
            self.fetch_data()
        
        return self.data.get(option_name, {})
    
    def search_options(self, search_term: str) -> List[str]:
        """
        Search for options containing the search term.
        
        Args:
            search_term: Term to search for in option names
            
        Returns:
            List of matching option names
        """
        if not self.data:
            self.fetch_data()
        
        search_lower = search_term.lower()
        return [key for key in self.data.keys() 
                if search_lower in key.lower() and key != '_meta']
    
    def explore_nested_structure(self, data: Dict = None, 
                                  level: int = 0, max_level: int = 2) -> None:
        """
        Recursively explore the nested structure of the JSON data.
        
        Args:
            data: Data to explore (defaults to root data)
            level: Current nesting level
            max_level: Maximum depth to explore
        """
        if data is None:
            data = self.data
        
        if level > max_level:
            return
        
        indent = "  " * level
        
        for key, value in data.items():
            if isinstance(value, dict):
                print(f"{indent}{key}: {{...}}")
                if 'children' in value:
                    self.explore_nested_structure(value['children'], level + 1, max_level)
            elif isinstance(value, list):
                print(f"{indent}{key}: [{len(value)} items]")
            else:
                print(f"{indent}{key}: {value}")
    
    def save_to_file(self, filename: str = "highcharts_data.json") -> None:
        """
        Save the fetched JSON data to a file.
        
        Args:
            filename: Name of the file to save to
        """
        if not self.data:
            self.fetch_data()
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(self.data, f, indent=2, ensure_ascii=False)
        
        print(f"Data saved to {filename}")


def main():
    """Main function demonstrating the parser usage."""
    
    # Initialize parser
    parser = HighchartsJSONParser()
    
    # Fetch data
    try:
        parser.fetch_data()
    except Exception as e:
        print(f"Failed to fetch data: {e}")
        return
    
    print("\n" + "="*60)
    print("METADATA")
    print("="*60)
    metadata = parser.get_metadata()
    pprint(metadata)
    
    print("\n" + "="*60)
    print("ALL TOP-LEVEL OPTIONS")
    print("="*60)
    options = parser.get_all_options()
    print(f"Found {len(options)} options:")
    for i, option in enumerate(options[:10], 1):  # Show first 10
        print(f"{i}. {option}")
    if len(options) > 10:
        print(f"... and {len(options) - 10} more")
    
    print("\n" + "="*60)
    print("EXAMPLE: ACCESSIBILITY OPTION")
    print("="*60)
    accessibility = parser.get_option('accessibility')
    if accessibility:
        print("\nAccessibility option structure:")
        if 'doclet' in accessibility:
            print(f"\nDescription: {accessibility['doclet'].get('description', 'N/A')[:200]}...")
        if 'children' in accessibility:
            print(f"\nChild options: {list(accessibility['children'].keys())[:5]}...")
    
    print("\n" + "="*60)
    print("SEARCH EXAMPLE: Options containing 'chart'")
    print("="*60)
    chart_options = parser.search_options('chart')
    print(f"Found {len(chart_options)} options:")
    for option in chart_options[:5]:
        print(f"  - {option}")
    
    # Optional: Save to file
    print("\n" + "="*60)
    save_option = input("\nSave data to file? (y/n): ").strip().lower()
    if save_option == 'y':
        parser.save_to_file()


if __name__ == "__main__":
    main()