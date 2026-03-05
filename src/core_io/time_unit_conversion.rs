pub fn example_func(left : u64, right : u64) -> u64 {
    return left + right;
}

pub fn second_func() -> f32 {
    return 0.05;
}

#[cfg(test)]
mod tests {
    use crate::core_io;

    #[test]
    fn example_test(){
        assert_eq!(5, core_io::example_func(2,3));
    }
}