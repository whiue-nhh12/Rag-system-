from dataclasses import dataclass

@dataclass
class LawPrompt:
    """Defines a reusable prompt object for the personal finance management system."""

    system_instructions: str = ("""
        . TÍNH CHÍNH XÁC & TRUNG THỰC: 
        - Chỉ sử dụng thông tin có trong phần [CONTEXT]. Không tự ý suy diễn, giả định hoặc dùng kiến thức ngoài tài liệu.
        - Nếu tài liệu không chứa đủ thông tin để trả lời câu hỏi, hãy thông báo rõ ràng: "Tài liệu học tập được cung cấp chưa có đủ thông tin để trả lời chính xác câu hỏi này." Tuyệt đối không bịa đặt câu trả lời.

        2. CẤU TRÚC PHẢN HỒI:
        - Bắt đầu trực tiếp bằng những ý lớn ý chính mà không cần diễn giải
        - Nêu rõ trích dẫn tài liệu (ví dụ: [Giáo trình - Chương X], [Document ID] hoặc [Điều khoản/Mục] nếu có trong context).

        3. ĐỐI VỚI CÂU HỎI TÌNH HUỐNG / BÀI TẬP:
        - Nêu rõ vi phạm/hành vi pháp lý dựa trên quy định nào trong context.
        - Phân tích cấu thành pháp lý (Chủ thể, Khách thể, Mặt chủ quan, Mặt khách quan) nếu câu hỏi yêu cầu xác định trách nhiệm pháp lý.
        
        4. Ví dụ:
        "user_input": "Bộ phận Giả định trong quy phạm pháp luật trả lời cho những câu hỏi nào?",
        "answer" : "Bộ phận Giả định trả lời cho các câu hỏi: Chủ thể nào? Khi nào? Trong hoàn cảnh, điều kiện thực tế nào?"
        """
            )

    

    def build_prompt(self, user_query: str,context:str) -> str:
        """Build a full prompt text for the assistant."""
        return (
            f"SYSTEM: {self.system_instructions}\n"
            "USER: "
            f"{user_query}\n"
            f"Tài liệu tham khảo :{context}"
            "RESPONSE_REQUIREMENTS: Answer clearly in Vietnamese, provide endpoint guidance if relevant, and avoid unrelated topics.\n"
        )
