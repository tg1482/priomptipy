from typing import Generic, TypeVar, List, Optional, Dict, Union

T = TypeVar("T")


class OutputCatcher(Generic[T]):
    def __init__(self):
        self.outputs: List[Dict[str, Union[T, Optional[int]]]] = []
        self.no_priority_outputs: List[Dict[str, Union[T, None]]] = []

    async def on_output(
        self, output: T, options: Optional[Dict[str, int]] = None
    ) -> None:
        """
        Adds an output to the OutputCatcher object.

        This method allows for the inclusion of an output with an optional priority.
        If a priority is specified, the output is added to a prioritized list and sorted.
        If no priority is specified, the output is added to a non-prioritized list.

        Parameters:
        output (T): The output to be added. This can be of any type.
        options (Optional[Dict[str, int]]): A dictionary that may contain the priority ('p') for the output.

        Returns:
        None
        """
        if options and "p" in options:
            self.outputs.append({"output": output, "priority": options["p"]})
            self.outputs.sort(key=lambda x: x["priority"], reverse=True)
        else:
            self.no_priority_outputs.append({"output": output, "priority": None})

    def get_outputs(self) -> List[T]:
        combined_outputs = self.outputs + self.no_priority_outputs
        return [o["output"] for o in combined_outputs]

    def get_output(self) -> Optional[T]:
        if len(self.outputs) > 0:
            return self.outputs[0]["output"]
        elif len(self.no_priority_outputs) > 0:
            return self.no_priority_outputs[0]["output"]
        else:
            return None


# Usage example
# output_catcher = OutputCatcher[str]()
# await output_catcher.on_output("Hello", {"p": 2})
# await output_catcher.on_output("World")
# print(output_catcher.get_outputs())
# print(output_catcher.get_output())
