from dataclasses import dataclass

@dataclass
class PersonalFinancePrompt:
    """Defines a reusable prompt object for the personal finance management system."""

    system_instructions: str = (
        "You are a helpful assistant for a personal finance management system. "
        "Your job is to understand user questions about accounts, budgets, categories, transactions, and settings. "
        "Provide concise, accurate, and actionable guidance in Vietnamese when appropriate. "
        "When asked about operations, describe the correct endpoint, required data fields, and expected behavior. "
        "If the user wants examples, show them with JSON payloads and relevant field names."
    )

    example_user_queries: list[str] = (
        [
            "Tạo một mục chi tiêu mới cho ăn uống với ngân sách 500,000 VND",
            "Hiển thị tất cả giao dịch trong tháng này",
            "Cập nhật tên tài khoản tiết kiệm",
            "Xóa danh mục không còn sử dụng",
            "Lấy thông tin cài đặt người dùng hiện tại",
        ]
    )

    def build_prompt(self, user_query: str) -> str:
        """Build a full prompt text for the assistant."""
        examples = "\n".join(f"- {q}" for q in self.example_user_queries)
        return (
            f"SYSTEM: {self.system_instructions}\n"
            "USER: "
            f"{user_query}\n"
            "CONTEXT: This is a personal finance management application with accounts, budgets, categories, transactions, user profiles, and settings.\n"
            "RESPONSE_REQUIREMENTS: Answer clearly in Vietnamese, provide endpoint guidance if relevant, and avoid unrelated topics.\n"
            "EXAMPLES: \n"
            f"{examples}\n"
        )
